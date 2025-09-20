import socket
import json
import threading
import time
import os
import re

# PDF extraction libraries
try:
    import PyPDF2
    import fitz  # PyMuPDF - alternative PDF reader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  Install PDF libraries: pip install PyPDF2 PyMuPDF")

class PDFReceiver:
    def __init__(self):
        self.port = 8080
        self.discovery_port = 8079
        self.sock = None
        self.discovery_sock = None
        self.my_ip = self.get_my_ip()
        
        # PDF and text processing
        self.pdf_files = []
        self.extracted_texts = {}
        self.training_samples = []
        
        # Find PDF files on startup
        self.find_local_pdfs()
        
    def get_my_ip(self):
        try:
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_sock.connect(("8.8.8.8", 80))
            ip = temp_sock.getsockname()[0]
            temp_sock.close()
            return ip
        except:
            return "127.0.0.1"
    
    def find_local_pdfs(self):
        """Find PDF files in current directory"""
        print("🔍 Searching for PDF files...")
        
        current_dir = os.getcwd()
        for file in os.listdir(current_dir):
            if file.lower().endswith('.pdf'):
                self.pdf_files.append(file)
                print(f"📄 Found PDF: {file}")
        
        if not self.pdf_files:
            print("❌ No PDF files found in current directory")
            print("💡 Place some PDF files in the same folder as this script")
        else:
            print(f"✅ Found {len(self.pdf_files)} PDF files")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using multiple methods"""
        if not PDF_AVAILABLE:
            return "PDF libraries not installed"
        
        text = ""
        print(f"📖 Extracting text from {pdf_path}...")
        
        # Method 1: Try PyMuPDF first (usually better)
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
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
        text = re.sub(r'[^\w\s.,!?;:]', '', text)  # Keep basic punctuation
        text = text.strip()
        
        return text
    
    def create_training_samples(self, text, sample_type="sentences"):
        """Convert text into training samples"""
        samples = []
        
        if sample_type == "sentences":
            # Split into sentences for classification/analysis
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Skip very short sentences
                    samples.append({
                        'text': sentence,
                        'length': len(sentence),
                        'word_count': len(sentence.split())
                    })
        
        elif sample_type == "word_pairs":
            # Create word pairs for next-word prediction
            words = text.split()
            for i in range(len(words) - 1):
                if len(words[i]) > 2 and len(words[i+1]) > 2:  # Skip short words
                    samples.append({
                        'input_word': words[i].lower(),
                        'next_word': words[i+1].lower(),
                        'context': ' '.join(words[max(0, i-2):i+2])
                    })
        
        return samples
    
    def process_all_pdfs(self):
        """Extract and process all PDF files"""
        print("\n📚 Processing all PDF files...")
        
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
                word_pairs = self.create_training_samples(clean_text, "word_pairs")
                
                self.training_samples.extend(sentences[:100])  # Limit to 100 samples per book
                
                print(f"📊 {pdf_file}:")
                print(f"   Raw text: {len(raw_text):,} chars")
                print(f"   Clean text: {len(clean_text):,} chars") 
                print(f"   Sentences: {len(sentences)}")
                print(f"   Word pairs: {len(word_pairs)}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
        
        print(f"\n✅ Total training samples: {len(self.training_samples)}")
    
    def start_discovery_service(self):
        """Let other laptops find us"""
        self.discovery_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.discovery_sock.bind(('0.0.0.0', self.discovery_port))
        
        print(f"🔍 Discovery service running at {self.my_ip}")
        
        while True:
            try:
                message, sender_address = self.discovery_sock.recvfrom(1024)
                
                if message.decode('utf-8') == "FIND_RECEIVERS":
                    # Send info about our data
                    response = {
                        'type': 'receiver_info',
                        'ip': self.my_ip,
                        'port': self.port,
                        'pdf_count': len(self.pdf_files),
                        'sample_count': len(self.training_samples),
                        'pdf_names': [os.path.basename(f) for f in self.pdf_files]
                    }
                    self.discovery_sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
                    print(f"📡 Told {sender_address} about our {len(self.pdf_files)} PDF files")
                    
            except:
                break
    
    def handle_training_request(self, request, sender_address):
        """Handle requests for training data or samples"""
        request_type = request.get('type')
        
        if request_type == 'get_sample_data':
            # Send some sample data for analysis
            sample_data = {
                'type': 'sample_response',
                'samples': self.training_samples[:10],  # First 10 samples
                'total_samples': len(self.training_samples),
                'pdf_info': {
                    'count': len(self.pdf_files),
                    'names': [os.path.basename(f) for f in self.pdf_files]
                }
            }
            
            response = json.dumps(sample_data)
            self.sock.sendto(response.encode('utf-8'), sender_address)
            print(f"📤 Sent sample data to {sender_address}")
            
        elif request_type == 'get_text_stats':
            # Send statistics about our text data
            total_chars = sum(len(text) for text in self.extracted_texts.values())
            total_words = sum(len(text.split()) for text in self.extracted_texts.values())
            
            stats = {
                'type': 'stats_response',
                'total_characters': total_chars,
                'total_words': total_words,
                'total_samples': len(self.training_samples),
                'pdf_files': len(self.pdf_files),
                'avg_sentence_length': sum(s['length'] for s in self.training_samples) / len(self.training_samples) if self.training_samples else 0
            }
            
            response = json.dumps(stats)
            self.sock.sendto(response.encode('utf-8'), sender_address)
            print(f"📊 Sent text statistics to {sender_address}")
    
    def start_receiver(self):
        """Main receiver loop"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        
        print(f"📱 PDF Receiver listening on {self.my_ip}:{self.port}")
        print("📚 Ready to share training data from PDF files...")
        
        while True:
            try:
                message, sender_address = self.sock.recvfrom(8192)
                
                try:
                    data = json.loads(message.decode('utf-8'))
                    
                    if isinstance(data, dict):
                        self.handle_training_request(data, sender_address)
                    else:
                        print(f"📨 Unknown message format from {sender_address}")
                        
                except json.JSONDecodeError:
                    text = message.decode('utf-8')
                    print(f"📨 Plain text: {text[:100]}...")
                    
            except KeyboardInterrupt:
                print("\n👋 PDF Receiver shutting down...")
                print(f"📊 Final stats:")
                print(f"   PDF files processed: {len(self.pdf_files)}")
                print(f"   Training samples created: {len(self.training_samples)}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        if self.sock:
            self.sock.close()
        if self.discovery_sock:
            self.discovery_sock.close()
    
    def run(self):
        # Process PDFs first
        if self.pdf_files:
            self.process_all_pdfs()
        
        # Start discovery service in background
        discovery_thread = threading.Thread(target=self.start_discovery_service)
        discovery_thread.daemon = True
        discovery_thread.start()
        
        # Start main receiver
        self.start_receiver()

if __name__ == "__main__":
    print("📚 PDF RECEIVER - Distributed ML Training")
    print("="*60)
    print("Features: PDF Text Extraction + Training Data Preparation")
    print("="*60)
    
    if not PDF_AVAILABLE:
        print("🚨 Please install required libraries:")
        print("   pip install PyPDF2 PyMuPDF")
        print("   Then place PDF files in this directory")
    
    receiver = PDFReceiver()
    receiver.run()