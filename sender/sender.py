import socket
import time

def start_sender():
    # Create a socket (like getting a phone)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Ask user for the other laptop's IP address
    print("🔍 First, we need to find the other laptop...")
    print("On the other laptop, the receiver should show its IP address")
    print("Or try these common ones:")
    print("- If both laptops are on same WiFi: probably 192.168.1.X")
    print("- If testing on same laptop: use 127.0.0.1")
    
    target_ip = input("\n💻 Enter the receiver laptop's IP address: ")
    target_port = 8080
    
    print(f"\n📡 Trying to connect to {target_ip}:{target_port}")
    
    # Send a message
    message = "Hello PIHUUUU"
    print(f"📤 Sending: '{message}'")
    
    try:
        # Send the message
        sock.sendto(message.encode('utf-8'), (target_ip, target_port))
        
        # Wait for reply (with timeout)
        sock.settimeout(5.0)  # Wait max 5 seconds
        reply, receiver_address = sock.recvfrom(1024)
        
        print(f"🎉 SUCCESS! Got reply: '{reply.decode('utf-8')}'")
        print(f"✅ Two laptops are now talking!")
        
    except socket.timeout:
        print("⏰ No reply received. Check if:")
        print("  - Receiver is running on the other laptop")
        print("  - IP address is correct")
        print("  - Both laptops are on same network")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    sock.close()

if __name__ == "__main__":
    start_sender()