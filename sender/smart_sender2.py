import socket
import json
import random
import time
import hashlib

class SmartSender:
    def __init__(self):
        self.discovery_port = 8079
        self.receivers = []
        
        # AI Learning for optimal chunk sizes
        self.chunk_performance = []  # {size: X, time: Y, success: True/False}
        self.optimal_chunk_size = 1000  # Start with this
        self.min_chunk_size = 100
        self.max_chunk_size = 2000
        
        # Transfer tracking
        self.active_transfers = {}
        
    def find_receivers(self):
        """Find receiver laptops automatically"""
        print("🔍 Searching for receiver laptops...")
        
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
                response, address = sock.recvfrom(1024)
                response_text = response.decode('utf-8')
                
                if response_text.startswith("RECEIVER_HERE:"):
                    parts = response_text.split(":")
                    receiver_ip = parts[1]
                    receiver_port = int(parts[2])
                    
                    if (receiver_ip, receiver_port) not in self.receivers:
                        self.receivers.append((receiver_ip, receiver_port))
                        print(f"✅ Found receiver at {receiver_ip}:{receiver_port}")
                        found_any = True
                        
            except socket.timeout:
                continue
            except:
                break
        
        sock.close()
        
        if not found_any:
            print("❌ No receivers found automatically.")
            manual_ip = input("🔧 Enter receiver IP manually (or press Enter to skip): ")
            if manual_ip.strip():
                self.receivers.append((manual_ip.strip(), 8080))
        
        return len(self.receivers) > 0
    
    def learn_optimal_chunk_size(self):
        """AI: Learn the best chunk size from past performance"""
        if len(self.chunk_performance) < 3:
            return self.optimal_chunk_size
        
        # Analyze recent performance (last 10 transfers)
        recent_performance = self.chunk_performance[-10:]
        
        # Find the chunk size with best time/size ratio
        successful_transfers = [p for p in recent_performance if p['success']]
        
        if not successful_transfers:
            return self.optimal_chunk_size
        
        # Calculate efficiency score: data_size / time
        best_score = 0
        best_size = self.optimal_chunk_size
        
        for perf in successful_transfers:
            score = perf['data_size'] / perf['time'] if perf['time'] > 0 else 0
            if score > best_score:
                best_score = score
                best_size = perf['chunk_size']
        
        # Don't change too drastically
        if best_size != self.optimal_chunk_size:
            change = (best_size - self.optimal_chunk_size) * 0.3  # 30% of the difference
            new_size = int(self.optimal_chunk_size + change)
            new_size = max(self.min_chunk_size, min(self.max_chunk_size, new_size))
            
            print(f"🧠 AI Learning: Chunk size {self.optimal_chunk_size} → {new_size}")
            self.optimal_chunk_size = new_size
        
        return self.optimal_chunk_size
    
    def send_chunked_data(self, numbers, receiver_ip, receiver_port):
        """Send large data in intelligent chunks"""
        print(f"\n📊 Sending {len(numbers)} numbers to {receiver_ip}:{receiver_port}")
        
        # AI decides chunk size
        chunk_size = self.learn_optimal_chunk_size()
        print(f"🧠 AI suggests chunk size: {chunk_size}")
        
        # Create chunks
        chunks = [numbers[i:i+chunk_size] for i in range(0, len(numbers), chunk_size)]
        transfer_id = hashlib.md5(f"{time.time()}_{receiver_ip}".encode()).hexdigest()[:8]
        
        print(f"📦 Splitting into {len(chunks)} chunks (ID: {transfer_id})")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(10.0)
        
        start_time = time.time()
        successful_chunks = 0
        
        # Track this transfer
        self.active_transfers[transfer_id] = {
            'total_chunks': len(chunks),
            'received_acks': set(),
            'response_chunks': {},
            'start_time': start_time
        }
        
        # Send all chunks
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            message = {
                'type': 'chunk',
                'transfer_id': transfer_id,
                'chunk_num': i,
                'total_chunks': len(chunks),
                'data': chunk
            }
            
            try:
                sock.sendto(json.dumps(message).encode('utf-8'), (receiver_ip, receiver_port))
                
                # Wait for acknowledgment
                ack_received = False
                ack_timeout = time.time() + 2.0  # 2 second timeout per chunk
                
                while time.time() < ack_timeout:
                    try:
                        sock.settimeout(0.1)
                        response, addr = sock.recvfrom(4096)
                        ack_data = json.loads(response.decode('utf-8'))
                        
                        if (ack_data.get('type') == 'chunk_ack' and 
                            ack_data.get('transfer_id') == transfer_id and
                            ack_data.get('chunk_num') == i):
                            ack_received = True
                            successful_chunks += 1
                            break
                            
                    except (socket.timeout, json.JSONDecodeError):
                        continue
                
                chunk_time = time.time() - chunk_start
                
                if ack_received:
                    print(f"📥 Chunk {i+1}/{len(chunks)} ✅ ({chunk_time:.3f}s)")
                else:
                    print(f"📥 Chunk {i+1}/{len(chunks)} ❌ (timeout)")
                
                # Small delay to prevent overwhelming
                time.sleep(0.005)
                
            except Exception as e:
                print(f"❌ Error sending chunk {i+1}: {e}")
        
        # Wait for response chunks
        print(f"⏳ Waiting for processed response...")
        response_data = self.wait_for_response(sock, transfer_id, receiver_ip, receiver_port)
        
        transfer_time = time.time() - start_time
        success = successful_chunks == len(chunks) and response_data is not None
        
        # Record performance for AI learning
        self.chunk_performance.append({
            'chunk_size': chunk_size,
            'data_size': len(numbers),
            'time': transfer_time,
            'success': success,
            'chunks_sent': len(chunks),
            'chunks_acked': successful_chunks
        })
        
        print(f"\n📈 Transfer Summary:")
        print(f"   Time: {transfer_time:.2f}s")
        print(f"   Success rate: {successful_chunks}/{len(chunks)} chunks")
        print(f"   Data integrity: {'✅' if success else '❌'}")
        
        if response_data:
            print(f"   Verification: Original sum {sum(numbers)} → New sum {sum(response_data)}")
            print(f"   Doubled correctly: {'✅' if sum(response_data) == sum(numbers) * 2 else '❌'}")
        
        sock.close()
        return success
    
    def wait_for_response(self, sock, transfer_id, receiver_ip, receiver_port):
        """Wait for chunked response from receiver"""
        response_id = f"response_{transfer_id}"
        response_chunks = {}
        total_response_chunks = None
        
        timeout = time.time() + 15.0  # 15 second timeout for response
        
        while time.time() < timeout:
            try:
                sock.settimeout(1.0)
                message, addr = sock.recvfrom(16384)
                data = json.loads(message.decode('utf-8'))
                
                if (data.get('type') == 'response_chunk' and 
                    data.get('transfer_id') == response_id):
                    
                    chunk_num = data['chunk_num']
                    total_response_chunks = data['total_chunks']
                    response_chunks[chunk_num] = data['data']
                    
                    print(f"📨 Response chunk {chunk_num+1}/{total_response_chunks}")
                    
                    # Check if we have all response chunks
                    if len(response_chunks) == total_response_chunks:
                        # Reconstruct response
                        full_response = []
                        for i in range(total_response_chunks):
                            if i in response_chunks:
                                full_response.extend(response_chunks[i])
                        
                        print(f"✅ Complete response received: {len(full_response)} numbers")
                        return full_response
                
            except (socket.timeout, json.JSONDecodeError):
                continue
            except Exception as e:
                print(f"❌ Response error: {e}")
                break
        
        print("⏰ Response timeout")
        return None
    
    def interactive_mode(self):
        """Interactive menu with smart features"""
        while True:
            print("\n" + "="*60)
            print("🧠 SMART SENDER - AI-Powered Data Transfer")
            print(f"🎯 Current optimal chunk size: {self.optimal_chunk_size}")
            print(f"📊 Performance history: {len(self.chunk_performance)} transfers")
            print("="*60)
            print("1️⃣  Send 100 numbers (small test)")
            print("2️⃣  Send 1,000 numbers (medium test)")
            print("3️⃣  Send 10,000 numbers (large test)")
            print("4️⃣  Send 50,000 numbers (stress test)")
            print("5️⃣  Custom amount")
            print("6️⃣  Performance analysis")
            print("7️⃣  Find receivers again")
            print("8️⃣  Quit")
            
            choice = input("Choose (1-8): ")
            
            if choice == "1":
                self.send_test_data(100)
            elif choice == "2":
                self.send_test_data(1000)
            elif choice == "3":
                self.send_test_data(10000)
            elif choice == "4":
                self.send_test_data(50000)
            elif choice == "5":
                try:
                    count = int(input("How many numbers? "))
                    self.send_test_data(count)
                except ValueError:
                    print("❌ Invalid number!")
            elif choice == "6":
                self.show_performance_analysis()
            elif choice == "7":
                self.receivers = []
                self.find_receivers()
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
    
    def send_test_data(self, count):
        """Send test data to all receivers"""
        if not self.receivers:
            print("❌ No receivers available")
            return
        
        numbers = [random.randint(1, 100) for _ in range(count)]
        
        for receiver_ip, receiver_port in self.receivers:
            self.send_chunked_data(numbers, receiver_ip, receiver_port)
    
    def show_performance_analysis(self):
        """Show AI learning analysis"""
        if not self.chunk_performance:
            print("📊 No performance data yet")
            return
        
        print("\n📈 PERFORMANCE ANALYSIS")
        print("="*50)
        
        successful = [p for p in self.chunk_performance if p['success']]
        failed = [p for p in self.chunk_performance if not p['success']]
        
        print(f"Total transfers: {len(self.chunk_performance)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.chunk_performance)*100:.1f}%)")
        print(f"Failed: {len(failed)}")
        
        if successful:
            avg_time = sum(p['time'] for p in successful) / len(successful)
            total_data = sum(p['data_size'] for p in successful)
            avg_throughput = sum(p['data_size']/p['time'] for p in successful) / len(successful)
            
            print(f"\nSuccessful Transfers:")
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Total data sent: {total_data:,} numbers")
            print(f"  Average throughput: {avg_throughput:.0f} numbers/second")
            
            # Show chunk size evolution
            print(f"\n🧠 AI Learning Progress:")
            for i, p in enumerate(self.chunk_performance[-5:]):
                status = "✅" if p['success'] else "❌"
                print(f"  Transfer {len(self.chunk_performance)-4+i}: size={p['chunk_size']}, time={p['time']:.2f}s {status}")

if __name__ == "__main__":
    print("🧠 SMART SENDER v3.0")
    print("="*50)
    print("Features: AI Chunk Learning + Performance Tracking + Reliability")
    
    sender = SmartSender()
    
    if sender.find_receivers():
        sender.interactive_mode()
    else:
        print("😢 Could not find any receivers to talk to")