import socket
import json
import time
import os
import re

# PDF extraction libraries
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  Install PDF libraries: pip install PyPDF2 PyMuPDF")

class PDFSender:
    def __init__(self):
        self.discovery_port = 8079
        self.receivers = []
        
        # PDF and text processing
        self.pdf_files = []
        self.extracted_texts = {}
        self.training_samples = []
        
        # Find our own PDF files
        self.find_local_pdfs()
        
    def find_local_pdfs(self):
        """Find PDF files in current directory"""
        print("🔍 Searching for local PDF files...")
        
        current_dir = os.getcwd()
        for file in os.listdir(current_dir):
            if file.lower().endswith('.pdf'):
                self.pdf_files.append(file)
                print(f"📄 Found PDF: {file}")
        
        if not self.pdf_files:
            print("❌ No PDF files found in current directory")
            print("💡 Place some PDF files in the same folder as this script")
        else:
            print(f"✅ Found {len(self.pdf_files)} PDF files locally")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using multiple methods"""
        if not PDF_AVAILABLE:
            return "PDF libraries not installed"
        
        text = ""
        print(f"📖 Extracting text from {pdf_path}...")
        
        # Method 1: Try PyMuPDF first
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(10, len(doc))):  # First 10 pages only
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            print(f"✅ Extracted using PyMuPDF: {len(text)} characters")
            
        except Exception as e:
            print(f"PyMuPDF failed: {e}")
            
            # Method 2: Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(min(10, len(reader.pages))):
                        page = reader.pages[page_num]
                        text += page.extract_text()
                print(f"✅ Extracted using PyPDF2: {len(text)} characters")
                
            except Exception as e2:
                print(f"❌ Both PDF methods failed: {e2}")
                return ""
        
        return text
    
    def clean_text(self, text):
        """Clean and prepare text for ML"""
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
        text = re.sub(r'[^\w\s.,!?;:]', '', text)  # Keep basic punctuation
        text = text.strip()
        return text
    
    def create_training_samples(self, text, sample_type="sentences"):
        """Convert text into training samples"""
        samples = []
        
        if sample_type == "sentences":
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    samples.append({
                        'text': sentence,
                        'length': len(sentence),
                        'word_count': len(sentence.split())
                    })
        
        elif sample_type == "word_pairs":
            words = text.split()
            for i in range(len(words) - 1):
                if len(words[i]) > 2 and len(words[i+1]) > 2:
                    samples.append({
                        'input_word': words[i].lower(),
                        'next_word': words[i+1].lower(),
                        'context': ' '.join(words[max(0, i-2):i+2])
                    })
        
        return samples
    
    def process_local_pdfs(self):
        """Process our own PDF files"""
        print("\n📚 Processing local PDF files...")
        
        for pdf_file in self.pdf_files:
            try:
                # Extract text
                raw_text = self.extract_text_from_pdf(pdf_file)
                if not raw_text:
                    continue
                
                # Clean text
                clean_text = self.clean_text(raw_text)
                self.extracted_texts[pdf_file] = clean_text
                
                # Create training samples
                sentences = self.create_training_samples(clean_text, "sentences")
                
                self.training_samples.extend(sentences[:100])  # Limit to 100 samples per book
                
                print(f"📊 {pdf_file}:")
                print(f"   Raw text: {len(raw_text):,} chars")
                print(f"   Clean text: {len(clean_text):,} chars")
                print(f"   Sentences: {len(sentences)}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
        
        print(f"\n✅ Our training samples: {len(self.training_samples)}")
    
    def find_receivers(self):
        """Find PDF receivers on network"""
        print("🔍 Searching for PDF receivers...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(3.0)
        
        broadcast_message = "FIND_RECEIVERS"
        networks = ["192.168.1.255", "192.168.0.255", "10.0.0.255", "127.0.0.1"]
        
        for network in networks:
            try:
                sock.sendto(broadcast_message.encode('utf-8'), (network, self.discovery_port))
            except:
                continue
        
        found_any = False
        start_time = time.time()
        
        while time.time() - start_time < 3.0:
            try:
                response, address = sock.recvfrom(4096)
                
                try:
                    receiver_info = json.loads(response.decode('utf-8'))
                    
                    if receiver_info.get('type') == 'receiver_info':
                        receiver_ip = receiver_info['ip']
                        receiver_port = receiver_info['port']
                        pdf_count = receiver_info.get('pdf_count', 0)
                        sample_count = receiver_info.get('sample_count', 0)
                        pdf_names = receiver_info.get('pdf_names', [])
                        
                        if (receiver_ip, receiver_port) not in [(r['ip'], r['port']) for r in self.receivers]:
                            self.receivers.append({
                                'ip': receiver_ip,
                                'port': receiver_port,
                                'pdf_count': pdf_count,
                                'sample_count': sample_count,
                                'pdf_names': pdf_names
                            })
                            
                            print(f"✅ Found receiver at {receiver_ip}:{receiver_port}")
                            print(f"   📚 Their PDFs: {pdf_count} files")
                            print(f"   📊 Their samples: {sample_count}")
                            print(f"   📄 Their books: {', '.join(pdf_names[:3])}{'...' if len(pdf_names) > 3 else ''}")
                            found_any = True
                
                except json.JSONDecodeError:
                    continue
                        
            except socket.timeout:
                continue
            except:
                break
        
        sock.close()
        
        if not found_any:
            print("❌ No PDF receivers found automatically.")
            manual_ip = input("🔧 Enter receiver IP manually (or press Enter to skip): ")
            if manual_ip.strip():
                self.receivers.append({
                    'ip': manual_ip.strip(),
                    'port': 8080,
                    'pdf_count': 0,
                    'sample_count': 0,
                    'pdf_names': []
                })
        
        return len(self.receivers) > 0
    
    def request_sample_data(self, receiver):
        """Request sample training data from receiver"""
        print(f"\n📤 Requesting sample data from {receiver['ip']}...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        
        request = {'type': 'get_sample_data'}
        
        try:
            sock.sendto(json.dumps(request).encode('utf-8'), (receiver['ip'], receiver['port']))
            
            response, addr = sock.recvfrom(8192)
            data = json.loads(response.decode('utf-8'))
            
            if data.get('type') == 'sample_response':
                samples = data['samples']
                total_samples = data['total_samples']
                pdf_info = data['pdf_info']
                
                print(f"📨 Received sample data:")
                print(f"   Sample count: {len(samples)} (out of {total_samples} total)")
                print(f"   Their books: {pdf_info['count']} files")
                
                # Show some sample content
                for i, sample in enumerate(samples[:3]):
                    text_preview = sample['text'][:60] + "..." if len(sample['text']) > 60 else sample['text']
                    print(f"   Sample {i+1}: \"{text_preview}\"")
                
                return samples
                
        except socket.timeout:
            print("⏰ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        sock.close()
        return []
    
    def request_text_stats(self, receiver):
        """Request text statistics from receiver"""
        print(f"\n📊 Requesting text statistics from {receiver['ip']}...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        
        request = {'type': 'get_text_stats'}
        
        try:
            sock.sendto(json.dumps(request).encode('utf-8'), (receiver['ip'], receiver['port']))
            
            response, addr = sock.recvfrom(4096)
            data = json.loads(response.decode('utf-8'))
            
            if data.get('type') == 'stats_response':
                print(f"📈 Statistics from {receiver['ip']}:")
                print(f"   Total characters: {data['total_characters']:,}")
                print(f"   Total words: {data['total_words']:,}")
                print(f"   Training samples: {data['total_samples']:,}")
                print(f"   PDF files: {data['pdf_files']}")
                print(f"   Avg sentence length: {data['avg_sentence_length']:.1f} chars")
                
                return data
                
        except socket.timeout:
            print("⏰ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        sock.close()
        return {}
    
    def compare_datasets(self):
        """Compare our data with receivers' data"""
        print("\n📊 DATASET COMPARISON")
        print("="*50)
        
        # Our stats
        our_chars = sum(len(text) for text in self.extracted_texts.values())
        our_words = sum(len(text.split()) for text in self.extracted_texts.values())
        
        print(f"📚 OUR DATA:")
        print(f"   PDF files: {len(self.pdf_files)}")
        print(f"   Characters: {our_chars:,}")
        print(f"   Words: {our_words:,}")
        print(f"   Training samples: {len(self.training_samples)}")
        
        # Get stats from all receivers
        for i, receiver in enumerate(self.receivers):
            print(f"\n📚 RECEIVER {i+1} ({receiver['ip']}):")
            stats = self.request_text_stats(receiver)
            
        print(f"\n🎯 COMBINED DATASET:")
        total_receivers = len(self.receivers)
        print(f"   Total nodes: {total_receivers + 1} (us + {total_receivers} receivers)")
        print(f"   This enables distributed machine learning!")
    
    def interactive_mode(self):
        """Interactive menu for PDF operations"""
        while True:
            print("\n" + "="*60)
            print("📚 PDF SENDER - Distributed ML Training")
            print(f"📄 Local PDFs: {len(self.pdf_files)} files")
            print(f"📊 Local samples: {len(self.training_samples)}")
            print(f"🌐 Connected receivers: {len(self.receivers)}")
            print("="*60)
            print("1️⃣  Show local PDF content")
            print("2️⃣  Request sample data from receivers")
            print("3️⃣  Compare all datasets")
            print("4️⃣  Analyze text patterns")
            print("5️⃣  Find receivers again")
            print("6️⃣  Quit")
            
            choice = input("Choose (1-6): ")
            
            if choice == "1":
                self.show_local_content()
            elif choice == "2":
                self.request_all_samples()
            elif choice == "3":
                self.compare_datasets()
            elif choice == "4":
                self.analyze_text_patterns()
            elif choice == "5":
                self.receivers = []
                self.find_receivers()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
    
    def show_local_content(self):
        """Show preview of our PDF content"""
        print("\n📄 LOCAL PDF CONTENT:")
        print("="*40)
        
        for pdf_file in self.pdf_files:
            if pdf_file in self.extracted_texts:
                text = self.extracted_texts[pdf_file]
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"\n📚 {pdf_file}:")
                print(f"   Length: {len(text):,} characters")
                print(f"   Preview: \"{preview}\"")
    
    def request_all_samples(self):
        """Request samples from all receivers"""
        if not self.receivers:
            print("❌ No receivers connected")
            return
        
        for receiver in self.receivers:
            samples = self.request_sample_data(receiver)
            time.sleep(1)  # Small delay between requests
    
    def analyze_text_patterns(self):
        """Simple text analysis across all data"""
        print("\n🔍 TEXT PATTERN ANALYSIS")
        print("="*40)
        
        if not self.training_samples:
            print("❌ No local training samples to analyze")
            return
        
        # Analyze our data
        word_counts = [sample['word_count'] for sample in self.training_samples]
        sentence_lengths = [sample['length'] for sample in self.training_samples]
        
        print(f"📊 Local Analysis:")
        print(f"   Avg words per sentence: {sum(word_counts)/len(word_counts):.1f}")
        print(f"   Avg sentence length: {sum(sentence_lengths)/len(sentence_lengths):.1f} chars")
        print(f"   Shortest sentence: {min(sentence_lengths)} chars")
        print(f"   Longest sentence: {max(sentence_lengths)} chars")
        
        # Show common words (simple analysis)
        all_words = []
        for sample in self.training_samples[:10]:  # First 10 samples
            all_words.extend(sample['text'].lower().split())
        
        word_freq = {}
        for word in all_words:
            word = word.strip('.,!?;:')
            if len(word) > 3:  # Only words longer than 3 chars
                word_freq[word] = word_freq.get(word, 0) + 1
        
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🔤 Most common words:")
        for word, count in common_words:
            print(f"   '{word}': {count} times")

if __name__ == "__main__":
    print("📤 PDF SENDER - Distributed ML Training")
    print("="*60)
    print("Features: PDF Text Extraction + Data Analysis + Network Discovery")
    print("="*60)
    
    if not PDF_AVAILABLE:
        print("🚨 Please install required libraries:")
        print("   pip install PyPDF2 PyMuPDF")
        print("   Then place PDF files in this directory")
    
    sender = PDFSender()
    
    # Process our PDFs first
    if sender.pdf_files:
        sender.process_local_pdfs()
    
    if sender.find_receivers():
        sender.interactive_mode()
    else:
        print("😢 Could not find any PDF receivers to connect to")
        print("💡 Make sure pdf_receiver.py is running on another laptop")