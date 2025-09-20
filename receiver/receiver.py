import socket
import time

# Create a "listener" that waits for messages
def start_receiver():
    # Create a socket (like opening a mailbox)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Listen on this computer's IP address, port 8080
    sock.bind(('0.0.0.0', 8080))
    
    print("🎧 Receiver is listening on port 8080...")
    print("Waiting for messages from other laptops...")
    
    while True:
        try:
            # Wait for a message to arrive
            message, sender_address = sock.recvfrom(1024)
            
            # Convert bytes back to text
            text_message = message.decode('utf-8')
            
            print(f"📨 Got message: '{text_message}' from {sender_address}")
            
            # Send a reply back
            reply = f"Hello back! I received: {text_message}"
            sock.sendto(reply.encode('utf-8'), sender_address)
            print(f"📤 Sent reply back to {sender_address}")
            
        except KeyboardInterrupt:
            print("\n👋 Receiver shutting down...")
            break
    
    sock.close()

if __name__ == "__main__":
    start_receiver()