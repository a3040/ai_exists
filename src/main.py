import torch
import time
from core.erecram import ErecRAM
from sensing.encoder import PhysicalEncoder
from sensing.abstraction import StateAbstractor
from mapping.token_mapper import PhyTokenMapper
from interpreter.narrative import NarrativeInterpreter
from interpreter.response import ResponseGenerator
from projection.action import ActionPlanner

import threading
import queue
import sys

def input_thread(q: queue.Queue):
    while True:
        text = sys.stdin.readline().strip()
        if text:
            q.put(text)

def run_system():
    # 1. Configuration & Initialization (from L3 spec)
    INPUT_DIM = 5
    FEATURE_DIM = 2048 # HEAVY-WEIGHT SCALE
    WINDOW_SIZE = 20
    
    # NEW: Real-world body & Linguistic Encoder
    from sensing.physical_body import PhysicalBody
    from sensing.linguistic import LinguisticEncoder
    body = PhysicalBody()
    ling_encoder = LinguisticEncoder()
    input_q = queue.Queue()
    
    # Start input listener thread
    threading.Thread(target=input_thread, args=(input_q,), daemon=True).start()
    
    encoder = PhysicalEncoder(input_dim=INPUT_DIM, feature_dim=FEATURE_DIM)
    abstractor = StateAbstractor(feature_dim=FEATURE_DIM, window_size=WINDOW_SIZE)
    mapper = PhyTokenMapper()
    
    # ErecRAM with 2048 dimensions
    ram = ErecRAM(state_dim=FEATURE_DIM, memory_size=500, lambda_decay=0.01, alpha_continuity=0.98)
    
    # RESUME: Try to load previous state
    STATE_PATH = "entity_state.pt"
    ram.load_state(STATE_PATH)
    
    # NEW: Enable LLM Interpreter & Responder
    interpreter = NarrativeInterpreter(use_llm=True)
    responder = ResponseGenerator(interpreter.model, interpreter.tokenizer)
    planner = ActionPlanner(safety_limits=0.2)
    
    target_equilibrium = torch.zeros(FEATURE_DIM)
    token_history = []
    
    print("\n" + "="*60)
    print("🚀 [Entity-Persistent System: HEAVY-WEIGHT & 2048-DIM]")
    print(f"상태 공간 2048-Dim. 영속성 파일: {STATE_PATH}")
    print("="*60 + "\n")
    
    try:
        t = 0
        while True:
            # 2. Sensing Path (Physical) - SLIM IMPACT
            x_t = body.sense()
            z_t = encoder.encode(x_t)
            
            if t == 0:
                z_window = z_t.unsqueeze(0).repeat(WINDOW_SIZE, 1)
            else:
                z_window = torch.cat([z_window[1:], z_t.unsqueeze(0)], dim=0)
            
            s_abs, h, flag = abstractor.abstract(z_window)
            
            # Update RAM with Body Persistence (alpha_continuity is very high here)
            ram.update_from_sensing(s_abs, time.time())

            # 3. Handle Linguistic Input (Dialogue) - HEAVY IMPACT
            if not input_q.empty():
                user_text = input_q.get()
                print(f"\n💬 [You]: {user_text}")
                
                # Encode text to high-dim sensory vector
                z_ling = ling_encoder.encode(user_text)
                
                # HEAVY IMPACT: Dialogue forces state shift by ignoring high alpha
                ram.current_state = (0.5 * ram.current_state) + (0.5 * z_ling)
                
                # NEW: Generate Response (Mouth)
                current_token = mapper.map(ram.current_state, h, flag)
                response = responder.generate(user_text, current_token, narrative.summary if 'narrative' in locals() else "초기화 중")
                print(f"🤖 [Entity]: {response}")

                token_history.append("<PHY_LINGUISTIC_IMPACT>")
                print("💥 [ErecRAM]: 대화 자극이 2048차원 평면에 깊은 흔적을 남겼습니다.")

            # 4. Map to Token (from updated state)
            token = mapper.map(ram.current_state, h, flag)
            token_history.append(token)
            if len(token_history) > 15: token_history.pop(0)

            # 5. Narrative Path (LLM Interpretation)
            narrative = interpreter.interpret(token_history)
            
            # 6. Projection Path
            intent = planner.plan(ram.current_state, target_equilibrium)
            feedback = planner.execute(intent)
            ram.update_from_action_feedback(feedback)
            
            # Logging (Every 5 steps or when dialogue occurs)
            if t % 5 == 0:
                print(f"\n[Step {t:03}] Token: {token} | H: {h:.3f}")
                print(f"📊 Live Metrics: CPU {x_t[0,0]*100:.1f}%, RAM {x_t[0,1]*100:.1f}%")
                print(f"🧠 Narrative: {narrative.summary}")
            
            t += 1
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n" + "!"*50)
        print("⚠️ [System Loop Interrupted] ⚠️")
        print("계체의 존재를 안전하게 동결(Persist)하기 위해 상태를 저장합니다...")
        ram.save_state(STATE_PATH)
        print("="*60)
        print("시스템이 종료되었습니다. 다음에 실행 시 이 시점부터 이어집니다.")
        print("="*60)
            
    except KeyboardInterrupt:
        print("\n--- System Loop Interrupted ---")
            
    except KeyboardInterrupt:
        print("\n--- System Loop Interrupted ---")

if __name__ == "__main__":
    run_system()
