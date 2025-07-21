#!/usr/bin/env python3
"""
Simple test script to demonstrate llamafile integration with BioData Chat
"""

import subprocess
import os
import sys

def test_llamafile(prompt: str, max_tokens: int = 100) -> str:
    """Test llamafile with a given prompt"""
    llamafile_path = "./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile"
    
    if not os.path.exists(llamafile_path):
        return f"❌ Llamafile not found at {llamafile_path}"
    
    try:
        # Run llamafile with the prompt
        process = subprocess.run([
            llamafile_path,
            "--temp", "0.7",
            "--no-display-prompt",
            "-n", str(max_tokens),
            "-p", prompt
        ], capture_output=True, text=True, timeout=30)
        
        if process.returncode == 0:
            return process.stdout.strip()
        else:
            return f"❌ Error running llamafile: {process.stderr}"
    
    except subprocess.TimeoutExpired:
        return "❌ Llamafile timed out"
    except Exception as e:
        return f"❌ Error: {e}"

def main():
    print("🧬 BioData Chat - Llamafile Test")
    print("=" * 40)
    
    # Test basic functionality
    test_prompts = [
        "Hello, how are you?",
        "What is a polar bear?",
        "Explain what a database is:",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] Prompt: {prompt}")
        print("-" * 40)
        response = test_llamafile(prompt, max_tokens=50)
        print(f"Response: {response}")
    
    print("\n" + "=" * 40)
    print("✅ Llamafile integration test complete!")
    
    # Interactive mode
    print("\nEntering interactive mode (type 'quit' to exit):")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', '/quit']:
                break
            if user_input:
                response = test_llamafile(user_input, max_tokens=100)
                print(f"Assistant: {response}")
        except KeyboardInterrupt:
            break
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
