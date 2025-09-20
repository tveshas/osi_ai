import socket
import json
import threading
import time

class SmartReceiver:
    def __init__(self):
        self.port = 8080
        self.discovery_port = 8079
        self.sock = None
        self.discovery_sock = None
        self.my_ip = self.get_my_ip()
        
    def get_my_ip(self):
        # Quick way to find our IP address
        try:
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_sock.connect(("8.8.8.8", 80))  # Connect to Google DNS
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
                    # Someone is looking for receivers, tell them about us!
                    response = f"RECEIVER_HERE:{self.my_ip}:{self.port}"
                    self.discovery_sock.sendto(response.encode('utf-8'), sender_address)
                    print(f"📡 Told {sender_address} that I'm available")
                    
            except:
                break
    
    def start_receiver(self):
        """Handle actual data messages"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.port))
        
        print(f"📱 Data receiver listening on {self.my_ip}:{self.port}")
        print("📊 Ready to receive numbers from other laptops...")
        
        while True:
            try:
                message, sender_address = self.sock.recvfrom(8192)  # Bigger buffer for large lists
                
                print(f"🔍 Raw message size: {len(message)} bytes")  # Debug info
                
                # Try to parse as JSON (list of numbers)
                try:
                    text = message.decode('utf-8')
                    print(f"🔍 Decoded text: {text[:100]}...")  # Show first 100 chars
                    
                    data = json.loads(text)
                    
                    if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                        # It's a list of numbers!
                        print(f"📊 Received {len(data)} numbers from {sender_address}")
                        
                        # Show more numbers for debugging
                        if len(data) <= 20:
                            print(f"   All numbers: {data}")
                        else:
                            print(f"   First 10: {data[:10]}")
                            print(f"   Last 10: {data[-10:]}")
                        
                        print(f"   Sum: {sum(data)}")
                        print(f"   Min: {min(data)}, Max: {max(data)}")
                        
                        # Send back some processed numbers
                        reply_numbers = [x * 2 for x in data]  # Double all numbers
                        reply_json = json.dumps(reply_numbers)
                        
                        print(f"🔍 Reply size: {len(reply_json)} bytes")  # Debug info
                        
                        self.sock.sendto(reply_json.encode('utf-8'), sender_address)
                        print(f"📤 Sent back {len(reply_numbers)} doubled numbers")
                        print(f"   Reply sum: {sum(reply_numbers)}")
                        
                    else:
                        # Regular text message
                        print(f"📨 Text message: '{data}' from {sender_address}")
                        
                except json.JSONDecodeError as e:
                    # Not JSON, treat as regular text
                    text = message.decode('utf-8')
                    print(f"❌ JSON Error: {e}")
                    print(f"📨 Text: '{text}' from {sender_address}")
                except Exception as e:
                    print(f"❌ Processing Error: {e}")
                    
            except KeyboardInterrupt:
                print("\n👋 Receiver shutting down...")
                break
            except Exception as e:
                print(f"❌ Socket Error: {e}")
        
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
    receiver = SmartReceiver()
    receiver.run()