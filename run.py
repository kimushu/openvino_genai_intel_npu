import os
import time
import openvino_genai as ov_genai
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.rule import Rule
from rich.text import Text
from rich.align import Align
from rich.table import Table
from huggingface_hub import snapshot_download

# CONFIG: Qwen3 Local Directory
MODEL_DIR = "Qwen3-8B-Instruct-NPU-Model"
THEME_COLOR = "color(33)" # Professional Blue
console = Console()

class QwenCLI:
    def __init__(self):
        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = 1024
        self.device = "NPU"
        self.chat_history = []
        self.perf_metrics = []
        self.pipe = None

    def print_logo(self):
        logo_text = r"""
  ___  __        __ _____  _   _     _____      _     ___ 
 / _ \ \ \      / /| ____|| \ | |   |___ /     / \   |_ _|
| | | | \ \ /\ / / |  _|  |  \| |     |_ \    / _ \   | | 
| |_| |  \ V  V /  | |___ | |\  |    ___) |  / ___ \  | | 
 \__\_\   \_/\_/   |_____||_| \_|   |____/  /_/   \_\|___|
"""
        styled_logo = Text(logo_text)
        styled_logo.stylize(f"bold {THEME_COLOR}")    
        
        console.print(Align.center(styled_logo))
        console.print(Align.center(f"[{THEME_COLOR}]Local Inference Engine[/{THEME_COLOR}]\n"))

    def show_metrics(self):
        if not self.perf_metrics:
            console.print(f"[{THEME_COLOR}]No performance data available yet.[/{THEME_COLOR}]")
            return

        table = Table(title="Performance History", border_style=THEME_COLOR)
        table.add_column("Query #", justify="center")
        table.add_column("Tokens", justify="center")
        table.add_column("TTFT (s)", justify="center")
        table.add_column("TPOT (ms)", justify="center")
        table.add_column("Speed (TPS)", justify="right", style="bold green")

        for i, m in enumerate(self.perf_metrics, 1):
            table.add_row(
                str(i), 
                str(m['tokens']), 
                f"{m['ttft']:.2f}", 
                f"{m['tpot']:.2f}", 
                f"{m['tps']:.2f}"
            )
        
        console.print(table)

    def setup(self):
        console.clear()
        
        prompt_text = f"[bold {THEME_COLOR}]Select Hardware[/bold {THEME_COLOR}] (NPU/GPU/CPU) [NPU]: "
        self.device = console.input(prompt_text).upper() or "NPU"
        
        console.print(f"[{THEME_COLOR}]Initializing Qwen3 on {self.device}...[/{THEME_COLOR}]")
        
        pipeline_args = {
            'models_path': MODEL_DIR,
            'device': self.device,
            'collect_perf_metrics': True
        }
        
        self.pipe = ov_genai.LLMPipeline(**pipeline_args)
        self.pipe.start_chat()
        
        console.clear()
        self.print_logo()
        console.print(Rule(f"Ready on {self.device}", style=THEME_COLOR))
        console.print(Align.center(f"[{THEME_COLOR} dim]Type :help for commands[/{THEME_COLOR} dim]\n"))

    def handle_command(self, text):
        cmd = text.lower().strip()
        if cmd == ":exit": return "break"
        elif cmd == ":clear":
            self.pipe.finish_chat()
            self.pipe.start_chat()
            self.chat_history = []
            self.perf_metrics = []
            console.clear()
            self.print_logo()
            return "continue"
        elif cmd == ":metrics":
            self.show_metrics()
            return "continue"
        elif cmd == ":save":
            fname = f"chat_{int(time.time())}.txt"
            with open(fname, "w", encoding="utf-8") as f:
                f.write("\n".join(self.chat_history))
            console.print(f"[{THEME_COLOR}]Saved to {fname}[/{THEME_COLOR}]")
            return "continue"
        elif cmd == ":help":
            console.print(Markdown("- `:metrics` - Show performance history\n- `:save` - Export chat\n- `:clear` - Reset history\n- `:exit` - Close"))
            return "continue"
        return None

    def run(self):
        while True:
            try:
                console.print("\n" + "─" * console.width, style=f"dim {THEME_COLOR}")
                user_input = console.input(f"[bold {THEME_COLOR}]You>[/bold {THEME_COLOR}] ").strip()
                
                if not user_input: continue
                action = self.handle_command(user_input)
                if action == "break": break
                if action == "continue": continue

                # Enforce a professional, text-only output behind the scenes
                system_instruction = " Answer the user clearly and professionally. Do not use any emojis."
                prompt = user_input + system_instruction

                self.chat_history.append(f"You: {user_input}")
                console.print(f"\n[bold {THEME_COLOR} reverse] QWEN 3_ [/bold {THEME_COLOR} reverse]")
                
                full_response = ""
                token_count = 0
                start_time = time.time()
                
                with Live("", refresh_per_second=15) as live:
                    def streamer(token):
                        nonlocal full_response, token_count
                        full_response += token
                        token_count += 1
                        
                        # Format <think> tags into clean markdown blockquotes
                        display_text = full_response.replace('<think>', '> **Thinking...**\n> ').replace('</think>', '\n\n---\n\n')
                        live.update(Markdown(display_text))
                        return False

                    # Pass the modified prompt to the model
                    self.pipe.generate(prompt, self.config, streamer)
                
                # Manual Performance Metrics Calculation
                duration = time.time() - start_time
                tps = token_count / duration if duration > 0 else 0
                tpot = (duration / token_count * 1000) if token_count > 0 else 0
                
                self.perf_metrics.append({
                    "tokens": token_count, 
                    "ttft": 0.0,  # TTFT requires deep engine hooks, defaulting to 0
                    "tpot": tpot, 
                    "tps": tps
                })
                
                console.print(f"[dim]Speed: {tps:.2f} TPS | TPOT: {tpot:.2f}ms[/dim]")
                
                # Remove the hidden instruction from the saved history
                self.chat_history.append(f"Qwen: {full_response}")

            except KeyboardInterrupt: 
                break

if __name__ == "__main__":
    cli = QwenCLI()
    cli.setup()
    cli.run()