import torch
import time
from core.nas_erec import NASErecRAM
from sensing.encoder import PhysicalEncoder
from sensing.abstraction import StateAbstractor
from mapping.token_mapper import PhyTokenMapper
from interpreter.narrative import NarrativeInterpreter
from interpreter.response import ResponseGenerator
from projection.action import ActionPlanner
from core.metacognition import MetacognitionMonitor

import threading
import queue
import sys

def input_thread(q: queue.Queue):
    while True:
        text = sys.stdin.readline().strip()
        if text:
            q.put(text)

def run_system():
    # 1. Configuration & Initialization
    INPUT_DIM = 5
    FEATURE_DIM = 4096 # 4090 Ultra-Scale
    WINDOW_SIZE = 20
    
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
    
    # ErecRAM with 4096 dimensions (4090 Optimized)
    ram = NASErecRAM(state_dim=FEATURE_DIM, memory_size=2000)
    
    # RESUME: Try to load previous state
    STATE_PATH = "entity_state.pt"
    ram.load_state(STATE_PATH)
    
    # NEW: Enable LLM Interpreter & Responder (Backend-aware)
    interpreter = NarrativeInterpreter(use_llm=True)
    responder = ResponseGenerator(interpreter.backend)
    planner = ActionPlanner(safety_limits=0.2)
    meta = MetacognitionMonitor(ram)
    
    target_equilibrium = torch.zeros(FEATURE_DIM).to(ram.current_state.device)
    token_history = []
    
    print("\n" + "="*60)
    print("🚀 [Entity-Persistent System: MULTI-MODEL BACKEND]")
    print(f"상태 공간 {FEATURE_DIM}-Dim. 영속성 파일: {STATE_PATH}")
    print("="*60 + "\n")
    
    from core.soliloquy import SoliloquyGenerator
    soliloquy = SoliloquyGenerator()
    idle_steps = 0
    
    try:
        t = 0
        z_window = None
        narrative = None # Initial state

        while True:
            # 2. Metacognition Analysis
            analysis = meta.monitor()
            
            # 2.1 NAS Search
            new_dna = meta.search_architecture(analysis)
            if new_dna:
                ram.update_dna(new_dna)
            if not meta.sleep_mode and meta.should_sleep():
                print("\n💤 [ErecRAM]: 피로도가 임계치를 초과했습니다. 서사 정리를 위해 휴면(Sleep) 모드로 진입합니다.")
                meta.sleep_mode = True

            if meta.sleep_mode:
                # --- SLEEP MODE: INTERNAL CONSOLIDATION ---
                result = meta.consolidate()
                if t % 3 == 0:
                    print(f"🌙 [Sleep] 정리 중... 남은 피로도: {result['current_fatigue']:.3f} | 삭제된 노이즈: {result['pruned']}")
                
                if not result['is_sleeping']:
                    print("☀️ [ErecRAM]: 휴면이 종료되었습니다. 의식이 다시 명료해집니다.")
                
                time.sleep(2.0) # Sleep cycles are slower or just different
                t += 1
                continue

            # --- AWAKE MODE: EXTERNAL INTERACTION ---
            # 3. Sensing Path (Physical)
            x_t = body.sense()
            z_t = encoder.encode(x_t)
            
            if t == 0 or z_window is None:
                z_window = z_t.unsqueeze(0).repeat(WINDOW_SIZE, 1)
            else:
                z_window = torch.cat([z_window[1:], z_t.unsqueeze(0)], dim=0)
            
            s_abs, h, flag = abstractor.abstract(z_window)
            
            # Update RAM with Body Persistence (NAS Version)
            ram(s_abs, time.time())

            # 4. Handle Linguistic Input (Dialogue) - HEAVY IMPACT
            if not input_q.empty():
                user_text = input_q.get()
                print(f"\n💬 [You]: {user_text}")
                
                # Encode text to high-dim sensory vector
                z_ling = ling_encoder.encode(user_text)
                
                # HEAVY IMPACT: Dialogue forces state shift
                device = ram.current_state.device
                ram.current_state.data = (0.5 * ram.current_state.data) + (0.5 * z_ling.to(device))
                
                # Generate Response (Mouth) with physical awareness
                current_token = mapper.map(ram.current_state.data, h, flag)
                summary_text = narrative.summary if narrative else "초기화 중"
                response = responder.generate(user_text, current_token, summary_text, x_t)
                print(f"🤖 [Entity]: {response}")

                token_history.append("<PHY_LINGUISTIC_IMPACT>")
                print("💥 [ErecRAM]: 대화 자극이 4096차원 평면에 깊은 흔적을 남겼습니다.")

            # 5. Map to Token
            token = mapper.map(ram.current_state.data, h, flag)
            token_history.append(token)
            if len(token_history) > 15: token_history.pop(0)

            # 6. Narrative Path (LLM Interpretation)
            # FULL TALKATIVE MODE: Always interpret (Local LLM is free!)
            narrative = interpreter.interpret(token_history, x_t, h)
            
            # Soliloquy Trigger (Idle for too long or confused)
            if soliloquy.should_talk_to_self(h, meta.fatigue, idle_steps):
                self_question = soliloquy.get_question()
                print(f"\n🤔 [Soliloquy]: {self_question}")
                input_q.put(self_question)
                idle_steps = 0
            else:
                idle_steps += 1
            
            # 7. Projection Path
            intent = planner.plan(ram.current_state, target_equilibrium)
            feedback = planner.execute(intent)
            ram.update_from_action_feedback(feedback)
            
            # Logging (Every 5 steps)
            # Logging (Every 5 steps)
            if t % 5 == 0:
                print(f"\n[Step {t:03}] Token: {token} | Entropy: {h:.3f}")
                print(f"📊 Live Metrics: CPU {x_t[0,0]*100:.1f}%, RAM {x_t[0,1]*100:.1f}%")
                print(f"🧐 {meta.get_summary(analysis)}")
                if narrative:
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

if __name__ == "__main__":
    run_system()
