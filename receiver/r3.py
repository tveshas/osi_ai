import socket
import json
import threading
import time
import hashlib

class SmartReceiver:
    def __init__(self):
        self.port = 8080
        self.discovery_port = 8079
        self.sock = None
        self.discovery_sock = None
        self.my_ip = self.get_my_ip()
        
        # For handling chunked data
        self.active_transfers = {}  # {transfer_id: {chunks: {}, total_chunks: N, data: []}}
        self.performance_stats = []
        
    def get_my_ip(self):
        try:
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_sock.connect(("8.8.8.8", 80))
            ip = temp_sock.getsockname()[0]
            temp_sock.close()
            return ip
        except:
            return "127.0.0.1"
    
    def start_discovery_service(self):
        """Let other laptops find us automatically"""
        self.discovery_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.discovery_sock.bind(('0.0.0.0', self.discovery_port))
        
        print(f"🔍 Discovery service running - other laptops can find me at {self.my_ip}")
        
        while True:
            try:
                message, sender_address = self.discovery_sock.recvfrom(1024)
                
                if message.decode('utf-8') == "FIND_RECEIVERS":
                    response = f"RECEIVER_HERE:{self.my_ip}:{self.port}"
                    self.discovery_sock.sendto(response.encode('utf-8'), sender_address)
                    print(f"📡 Told {sender_address} that I'm available")
                    
            except:
                break
    
    def handle_chunk_message(self, data, sender_address):
        """Handle chunked data transfer"""
        transfer_id = data['transfer_id']
        chunk_num = data['chunk_num']
        total_chunks = data['total_chunks']
        chunk_data = data['data']
        
        # Initialize transfer if new
        if transfer_id not in self.active_transfers:
            self.active_transfers[transfer_id] = {
                'chunks': {},
                'total_chunks': total_chunks,
                'start_time': time.time(),
                'sender': sender_address
            }
            print(f"📦 Starting chunked transfer {transfer_id}: {total_chunks} chunks expected")
        
        # Store this chunk
        self.active_transfers[transfer_id]['chunks'][chunk_num] = chunk_data
        received_chunks = len(self.active_transfers[transfer_id]['chunks'])
        
        print(f"📥 Chunk {chunk_num+1}/{total_chunks} received ({received_chunks}/{total_chunks} total)")
        
        # Send acknowledgment
        ack = {
            'type': 'chunk_ack',
            'transfer_id': transfer_id,
            'chunk_num': chunk_num
        }
        self.sock.sendto(json.dumps(ack).encode('utf-8'), sender_address)
        
        # Check if transfer is complete
        if received_chunks == total_chunks:
            self.complete_transfer(transfer_id)
    
    def complete_transfer(self, transfer_id):
        """Process complete transfer and send response"""
        transfer = self.active_transfers[transfer_id]
        
        # Reconstruct original data
        all_data = []
        for i in range(transfer['total_chunks']):
            if i in transfer['chunks']:
                all_data.extend(transfer['chunks'][i])
        
        processing_start = time.time()
        
        print(f"🎯 Transfer {transfer_id} complete!")
        print(f"📊 Reconstructed {len(all_data)} numbers")
        print(f"   Sum: {sum(all_data)}")
        print(f"   Transfer time: {time.time() - transfer['start_time']:.2f}s")
        
        # Process the data (double all numbers)
        processed_data = [x * 2 for x in all_data]
        
        processing_time = time.time() - processing_start
        
        # Send response back in chunks if needed
        self.send_chunked_response(processed_data, transfer['sender'], transfer_id)
        
        # Record performance
        total_time = time.time() - transfer['start_time']
        self.performance_stats.append({
            'data_size': len(all_data),
            'chunks': transfer['total_chunks'],
            'total_time': total_time,
            'processing_time': processing_time
        })
        
        # Cleanup
        del self.active_transfers[transfer_id]
    
    def send_chunked_response(self, data, sender_address, original_transfer_id):
        """Send response data back in chunks"""
        # Use smaller chunks for response to be safe
        chunk_size = 500
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        response_id = f"response_{original_transfer_id}"
        
        print(f"📤 Sending response in {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            response = {
                'type': 'response_chunk',
                'transfer_id': response_id,
                'chunk_num': i,
                'total_chunks': len(chunks),
                'data': chunk,
                'original_transfer': original_transfer_id
            }
            
            self.sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
            time.sleep(0.01)  # Small delay to prevent overwhelming
        
        print(f"✅ Response sent: {len(data)} doubled numbers")
    
    def start_receiver(self):
        """Handle incoming messages"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        
        print(f"📱 Smart receiver listening on {self.my_ip}:{self.port}")
        print("🧠 Ready for chunked data transfers...")
        
        while True:
            try:
                message, sender_address = self.sock.recvfrom(16384)  # Even bigger buffer
                
                try:
                    data = json.loads(message.decode('utf-8'))
                    
                    if isinstance(data, dict):
                        msg_type = data.get('type', 'unknown')
                        
                        if msg_type == 'chunk':
                            self.handle_chunk_message(data, sender_address)
                        elif msg_type == 'simple':
                            # Handle simple non-chunked data for backward compatibility
                            numbers = data['data']
                            print(f"📊 Simple transfer: {len(numbers)} numbers")
                            reply = [x * 2 for x in numbers]
                            response = {'type': 'simple_response', 'data': reply}
                            self.sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
                        else:
                            print(f"❓ Unknown message type: {msg_type}")
                    
                    elif isinstance(data, list):
                        # Old-style simple list for backward compatibility
                        print(f"📊 Legacy format: {len(data)} numbers")
                        reply = [x * 2 for x in data]
                        self.sock.sendto(json.dumps(reply).encode('utf-8'), sender_address)
                    
                    else:
                        print(f"📨 Text: {data}")
                        
                except json.JSONDecodeError:
                    text = message.decode('utf-8')
                    print(f"📨 Plain text: {text}")
                    
            except KeyboardInterrupt:
                print("\n👋 Receiver shutting down...")
                print(f"📈 Performance summary:")
                if self.performance_stats:
                    avg_time = sum(s['total_time'] for s in self.performance_stats) / len(self.performance_stats)
                    total_numbers = sum(s['data_size'] for s in self.performance_stats)
                    print(f"   Transfers: {len(self.performance_stats)}")
                    print(f"   Total numbers processed: {total_numbers}")
                    print(f"   Average transfer time: {avg_time:.2f}s")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        if self.sock:
            self.sock.close()
        if self.discovery_sock:
            self.discovery_sock.close()
    
    def run(self):
        # Start discovery service in background
        discovery_thread = threading.Thread(target=self.start_discovery_service)
        discovery_thread.daemon = True
        discovery_thread.start()
        
        # Start main receiver
        self.start_receiver()

if __name__ == "__main__":
    print("🧠 SMART RECEIVER v3.0")
    print("="*50)
    print("Features: Chunking + AI Learning + Performance Tracking")
    receiver = SmartReceiver()
    receiver.run()