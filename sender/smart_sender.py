import socket
import json
import random
import time

class SmartSender:
    def __init__(self):
        self.discovery_port = 8079
        self.receivers = []
        
    def find_receivers(self):
        """Automatically find other laptops running receivers"""
        print("🔍 Searching for receiver laptops...")
        
        # Create a broadcast socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(3.0)  # Wait 3 seconds for responses
        
        # Send broadcast message to find receivers
        broadcast_message = "FIND_RECEIVERS"
        
        # Try common network ranges
        networks = [
            "192.168.1.255",    # Common home WiFi
            "192.168.0.255",    # Another common range  
            "10.0.0.255",       # Corporate networks
            "127.0.0.1"         # Local testing
        ]
        
        for network in networks:
            try:
                sock.sendto(broadcast_message.encode('utf-8'), (network, self.discovery_port))
            except:
                continue
        
        # Also try localhost for testing
        try:
            sock.sendto(broadcast_message.encode('utf-8'), ("127.0.0.1", self.discovery_port))
        except:
            pass
        
        # Listen for responses
        found_any = False
        start_time = time.time()
        
        while time.time() - start_time < 3.0:  # Listen for 3 seconds
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
            print("💡 Make sure receiver is running on other laptop!")
            manual_ip = input("🔧 Or enter receiver IP manually (press Enter to skip): ")
            if manual_ip.strip():
                self.receivers.append((manual_ip.strip(), 8080))
        
        return len(self.receivers) > 0
    
    def generate_numbers(self, count=10):
        """Generate a list of random numbers"""
        return [random.randint(1, 100) for _ in range(count)]
    
    def send_numbers(self):
        """Send numbers to all found receivers"""
        if not self.receivers:
            print("❌ No receivers available")
            return
        
        # Generate some numbers to send
        numbers = self.generate_numbers(10)
        print(f"📊 Generated numbers: {numbers}")
        print(f"📈 Sum: {sum(numbers)}, Average: {sum(numbers)/len(numbers):.1f}")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        
        for receiver_ip, receiver_port in self.receivers:
            try:
                print(f"\n📤 Sending to {receiver_ip}:{receiver_port}")
                
                # Send numbers as JSON
                message_json = json.dumps(numbers)
                print(f"🔍 Sending JSON: {message_json[:100]}...")  # Debug info
                print(f"🔍 Message size: {len(message_json)} bytes")
                
                sock.sendto(message_json.encode('utf-8'), (receiver_ip, receiver_port))
                
                # Wait for reply
                reply, address = sock.recvfrom(8192)  # Bigger buffer
                print(f"🔍 Reply size: {len(reply)} bytes")  # Debug info
                
                reply_json = reply.decode('utf-8')
                print(f"🔍 Reply JSON: {reply_json[:100]}...")  # Debug info
                
                reply_numbers = json.loads(reply_json)
                
                print(f"📨 Got back {len(reply_numbers)} numbers")
                
                # Show more details
                if len(reply_numbers) <= 20:
                    print(f"   All doubled numbers: {reply_numbers}")
                else:
                    print(f"   First 10 doubled: {reply_numbers[:10]}")
                    print(f"   Last 10 doubled: {reply_numbers[-10:]}")
                
                print(f"   Original sum: {sum(numbers)}, New sum: {sum(reply_numbers)}")
                print(f"   Verification: {sum(reply_numbers)} = {sum(numbers)} × 2? {sum(reply_numbers) == sum(numbers) * 2}")
                
            except socket.timeout:
                print(f"⏰ {receiver_ip} didn't respond in time")
            except json.JSONDecodeError as e:
                print(f"❌ JSON Error: {e}")
                print(f"🔍 Raw reply: {reply}")
            except Exception as e:
                print(f"❌ Error with {receiver_ip}: {e}")
        
        sock.close()
    
    def interactive_mode(self):
        """Let user send different amounts of numbers"""
        while True:
            print("\n" + "="*50)
            print("🎮 What do you want to send?")
            print("1️⃣  Send 10 random numbers")
            print("2️⃣  Send 100 random numbers") 
            print("3️⃣  Send 1000 random numbers")
            print("4️⃣  Find receivers again")
            print("5️⃣  Quit")
            
            choice = input("Choose (1-5): ")
            
            if choice == "1":
                numbers = self.generate_numbers(10)
                self.send_numbers_list(numbers)
            elif choice == "2":
                numbers = self.generate_numbers(100)
                self.send_numbers_list(numbers)
            elif choice == "3":
                numbers = self.generate_numbers(1000)
                self.send_numbers_list(numbers)
            elif choice == "4":
                self.receivers = []
                self.find_receivers()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
    
    def send_numbers_list(self, numbers):
        """Send a specific list of numbers"""
        if not self.receivers:
            print("❌ No receivers available")
            return
            
        print(f"📊 Sending {len(numbers)} numbers...")
        print(f"📈 Sum: {sum(numbers)}")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        
        for receiver_ip, receiver_port in self.receivers:
            try:
                message_json = json.dumps(numbers)
                sock.sendto(message_json.encode('utf-8'), (receiver_ip, receiver_port))
                
                reply, address = sock.recvfrom(8192)  # Bigger buffer
                reply_json = reply.decode('utf-8')
                reply_numbers = json.loads(reply_json)
                
                print(f"✅ {receiver_ip} processed {len(reply_numbers)} numbers")
                print(f"   Sum check: {sum(numbers)} → {sum(reply_numbers)} (doubled)")
                
            except Exception as e:
                print(f"❌ Error with {receiver_ip}: {e}")
        
        sock.close()

if __name__ == "__main__":
    sender = SmartSender()
    
    if sender.find_receivers():
        sender.interactive_mode()
    else:
        print("😢 Could not find any receivers to talk to")