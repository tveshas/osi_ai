import socket
import json
import random
import math
import time

class SimpleFederatedReceiver:
    def __init__(self):
        self.port = 8080
        self.discovery_port = 8079
        self.my_ip = self.get_my_ip()
        
        # Simple neural network - just 3 weights and 1 bias
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.01
        
        # Simple training data
        self.training_data = self.generate_training_data()
        self.training_rounds = 0
        
        print("🧠 Simple Federated Receiver")
        print(f"📍 IP: {self.my_ip}:{self.port}")
        print(f"🎯 Initial weights: {[round(w, 3) for w in self.weights]}")
        print(f"🎯 Initial bias: {round(self.bias, 3)}")
        print(f"📊 Training data: {len(self.training_data)} examples")
    
    def get_my_ip(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            sock.close()
            return ip
        except:
            return "127.0.0.1"
    
    def generate_training_data(self):
        """Generate simple training data: predict sum of 3 numbers"""
        data = []
        for _ in range(50):  # Just 50 examples
            inputs = [random.uniform(0, 5) for _ in range(3)]
            target = sum(inputs)  # Simple: just sum the inputs
            data.append({"inputs": inputs, "target": target})
        
        print(f"📚 Generated training data example:")
        print(f"   Input: {[round(x, 2) for x in data[0]['inputs']]} → Target: {round(data[0]['target'], 2)}")
        return data
    
    def forward_pass(self, inputs):
        """Simple neural network: output = sum(weights * inputs) + bias"""
        result = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return max(0, result)  # Simple ReLU activation
    
    def calculate_gradients(self):
        """Calculate gradients from our local data"""
        total_gradients = {"weights": [0, 0, 0], "bias": 0}
        total_loss = 0
        batch_size = 10  # Use 10 examples
        
        batch = random.sample(self.training_data, min(batch_size, len(self.training_data)))
        
        for example in batch:
            inputs = example["inputs"]
            target = example["target"]
            
            # Forward pass
            prediction = self.forward_pass(inputs)
            
            # Calculate loss (mean squared error)
            loss = (prediction - target) ** 2
            total_loss += loss
            
            # Simple gradient calculation
            error = prediction - target
            for i in range(3):
                total_gradients["weights"][i] += error * inputs[i] / batch_size
            total_gradients["bias"] += error / batch_size
        
        avg_loss = total_loss / batch_size
        return total_gradients, avg_loss
    
    def apply_gradients(self, gradients):
        """Apply gradients to update our model"""
        for i in range(3):
            self.weights[i] -= self.learning_rate * gradients["weights"][i]
        self.bias -= self.learning_rate * gradients["bias"]
    
    def federated_learning_step(self, received_data):
        """Do one step of federated learning"""
        received_gradients = received_data["gradients"]
        received_loss = received_data["loss"]
        
        print(f"\n🤝 Federated Learning Step {self.training_rounds + 1}")
        print(f"📨 Received gradients: {[round(g, 4) for g in received_gradients['weights']]}")
        print(f"📨 Received loss: {received_loss:.3f}")
        
        # Calculate our local gradients
        local_gradients, local_loss = self.calculate_gradients()
        print(f"💻 Our gradients: {[round(g, 4) for g in local_gradients['weights']]}")
        print(f"💻 Our loss: {local_loss:.3f}")
        
        # Average the gradients (federated learning!)
        combined_gradients = {
            "weights": [
                (local_gradients["weights"][i] + received_gradients["weights"][i]) / 2
                for i in range(3)
            ],
            "bias": (local_gradients["bias"] + received_gradients["bias"]) / 2
        }
        
        print(f"🔄 Combined gradients: {[round(g, 4) for g in combined_gradients['weights']]}")
        
        # Apply combined gradients
        old_weights = self.weights.copy()
        self.apply_gradients(combined_gradients)
        
        print(f"⚖️ Weights: {[round(w, 3) for w in old_weights]} → {[round(w, 3) for w in self.weights]}")
        
        self.training_rounds += 1
        
        # Send our gradients back
        response = {
            "gradients": local_gradients,
            "loss": local_loss,
            "round": self.training_rounds
        }
        
        return response
    
    def test_model(self):
        """Test our current model"""
        test_batch = random.sample(self.training_data, 5)
        total_error = 0
        
        print(f"\n🧪 Testing model:")
        for i, example in enumerate(test_batch):
            prediction = self.forward_pass(example["inputs"])
            actual = example["target"]
            error = abs(prediction - actual)
            total_error += error
            
            print(f"   Test {i+1}: {[round(x, 1) for x in example['inputs']]} → "
                  f"predicted {prediction:.2f}, actual {actual:.2f}, error {error:.2f}")
        
        avg_error = total_error / len(test_batch)
        print(f"📊 Average error: {avg_error:.3f}")
        return avg_error
    
    def start_discovery(self):
        """Simple discovery service"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.discovery_port))
        sock.settimeout(1.0)
        
        print(f"🔍 Discovery service running...")
        
        while True:
            try:
                message, sender_address = sock.recvfrom(1024)
                
                if message.decode('utf-8') == "FIND_FEDERATED_NODES":
                    response = {
                        "type": "federated_node",
                        "ip": self.my_ip,
                        "port": self.port,
                        "training_rounds": self.training_rounds
                    }
                    sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
                    print(f"📡 Told {sender_address[0]} about our node")
                    
            except socket.timeout:
                continue
            except:
                break
        
        sock.close()
    
    def start_receiver(self):
        """Main receiver loop"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.port))
        
        print(f"📱 Federated receiver listening on {self.my_ip}:{self.port}")
        print("🤝 Ready for federated learning...")
        
        # Test initial model
        self.test_model()
        
        while True:
            try:
                message, sender_address = sock.recvfrom(4096)
                
                try:
                    data = json.loads(message.decode('utf-8'))
                    
                    if data.get("type") == "federated_request":
                        print(f"\n🚀 Federated learning request from {sender_address[0]}")
                        
                        # Do federated learning
                        response = self.federated_learning_step(data)
                        
                        # Send response back
                        response["type"] = "federated_response"
                        sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
                        print(f"📤 Sent response back to {sender_address[0]}")
                        
                        # Test model after update
                        self.test_model()
                        
                    elif data.get("type") == "model_test":
                        # Just send our current model performance
                        error = self.test_model()
                        response = {
                            "type": "model_stats",
                            "error": error,
                            "weights": self.weights,
                            "bias": self.bias,
                            "rounds": self.training_rounds
                        }
                        sock.sendto(json.dumps(response).encode('utf-8'), sender_address)
                        
                    else:
                        print(f"❓ Unknown message: {data.get('type')}")
                        
                except json.JSONDecodeError:
                    print(f"📨 Non-JSON message: {message.decode('utf-8')[:50]}")
                    
            except KeyboardInterrupt:
                print(f"\n👋 Shutting down...")
                print(f"🏆 Final results:")
                print(f"   Training rounds: {self.training_rounds}")
                print(f"   Final weights: {[round(w, 3) for w in self.weights]}")
                print(f"   Final bias: {round(self.bias, 3)}")
                self.test_model()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        sock.close()
    
    def run(self):
        """Start everything"""
        import threading
        
        # Start discovery in background
        discovery_thread = threading.Thread(target=self.start_discovery)
        discovery_thread.daemon = True
        discovery_thread.start()
        
        # Start main receiver
        self.start_receiver()

if __name__ == "__main__":
    print("🤝 SIMPLE FEDERATED LEARNING - RECEIVER")
    print("="*50)
    print("Goal: Train neural network across two laptops")
    print("Method: Share gradients, not data")
    print("="*50)
    
    receiver = SimpleFederatedReceiver()
    receiver.run()