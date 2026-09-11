import asyncio
import time
import queue
import threading
from typing import List, Tuple, Any
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

try:
    import nexus_arena
    import nexus_weights_shm
    import nexus_nccl_fabric
    import nexus_health_monitor
except ImportError as e:
    print(f"[Warning] Native C++ extensions not pre-compiled: {e}")

app = FastAPI(title="Project Nexus Core Inference Engine", version="1.0.0")

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7

class BatchScheduler:
    def __init__(self, device_id: int = 0, max_batch_size: int = 32):
        self.device_id = device_id
        self.max_batch_size = max_batch_size
        self.request_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        
        try:
            self.health_monitor = nexus_health_monitor.NexusHealthMonitor(device_id, 3, 5000)
        except Exception:
            self.health_monitor = None

    def start(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.worker_thread.start()
        print("[NexusScheduler] Continuous dynamic batching engine started.")

    def stop(self):
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()
        print("[NexusScheduler] Engine stopped.")

    def submit(self, prompt: str, max_tokens: int) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        self.request_queue.put((prompt, max_tokens, future))
        return future

    def _execution_loop(self):
        while self.is_running:
            if self.health_monitor:
                state = self.health_monitor.get_circuit_state()
                if state == "OPEN":
                    time.sleep(0.1)
                    continue
                elif not self.health_monitor.check_hardware_health():
                    time.sleep(0.1)
                    continue

            batch: List[Tuple[str, int, asyncio.Future]] = []
            try:
                while len(batch) < self.max_batch_size:
                    block = self.request_queue.get_nowait()
                    batch.append(block)
            except queue.Empty:
                if not batch:
                    time.sleep(0.005)
                    continue

            try:
                results = self._process_batch_forward(batch)
                for (prompt, max_tokens, future), result in zip(batch, results):
                    if not future.done():
                        loop = future.get_loop()
                        loop.call_soon_threadsafe(future.set_result, result)
                
                if self.health_monitor:
                    self.health_monitor.record_success()

            except Exception as e:
                if self.health_monitor:
                    self.health_monitor.record_failure()
                for _, _, future in batch:
                    if not future.done():
                        loop = future.get_loop()
                        loop.call_soon_threadsafe(future.set_exception, e)

    def _process_batch_forward(self, batch: List[Tuple[str, int, Any]]) -> List[str]:
        time.sleep(0.015)
        return [f"Nexus engine processed: '{p}' (tokens: {m})" for p, m, _ in batch]

scheduler = BatchScheduler(device_id=0, max_batch_size=32)

@app.on_event("startup")
async def startup_event():
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.stop()

@app.post("/v1/generate")
async def generate_inference(req: InferenceRequest):
    if scheduler.health_monitor and scheduler.health_monitor.get_circuit_state() == "OPEN":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU circuit breaker is OPEN. Node is isolating hardware faults."
        )
    future = scheduler.submit(req.prompt, req.max_tokens)
    result = await future
    return {"status": "success", "output": result}

if __name__ == "__main__":
    uvicorn.run("nexus_orchestrator:app", host="0.0.0.0", port=8000, workers=1)