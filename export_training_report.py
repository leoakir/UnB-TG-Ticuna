"""
Exporta o histórico completo do treino direto do trainer_state.json, que tem
todos os registros de loss de treino e de avaliação, do passo 0 ao fim.

Gera:
  - training_history.csv   (todos os pontos registrados: step, epoch, loss,
                             grad_norm, learning_rate, eval_loss, eval_chrf++)
  - training_history.png   (gráfico com 4 painéis: loss, chrF++, grad_norm,
                             learning_rate)

Executar:
    python export_training_report.py
    python export_training_report.py --checkpoint nllb_tca_checkpoints/final
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def find_latest_checkpoint():
    checkpoints = sorted(
        config.MODEL_OUTPUT_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    final = config.MODEL_OUTPUT_DIR / "final"
    if final.exists():
        return final
    if not checkpoints:
        raise SystemExit(f"Nenhum checkpoint encontrado em {config.MODEL_OUTPUT_DIR}")
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Padrão: nllb_tca_checkpoints/final; se não existir, "
                              "usa o checkpoint-N mais recente.")
    parser.add_argument("--out-dir", type=str, default=".",
                         help="Pasta onde salvar o CSV e o PNG.")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    state_path = ckpt_dir / "trainer_state.json"
    print(f"Reading: {state_path}")
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    log_history = state["log_history"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CSV: todos os pontos registrados, de treino e de avaliação ----
    csv_path = out_dir / "training_history.csv"
    fieldnames = ["step", "epoch", "loss", "grad_norm", "learning_rate",
                  "eval_loss", "eval_chrf++", "eval_runtime"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in log_history:
            row = {k: entry.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"{len(log_history)} linhas gravadas em {csv_path}")

    # ---- Separa as séries de treino e de avaliação para o gráfico ----
    train_steps, train_loss, train_gradnorm, train_lr = [], [], [], []
    eval_steps, eval_loss_vals, eval_chrf_vals = [], [], []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
            train_gradnorm.append(entry.get("grad_norm"))
            train_lr.append(entry.get("learning_rate"))
        elif "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss_vals.append(entry["eval_loss"])
            eval_chrf_vals.append(entry.get("eval_chrf++"))

    # ---- Gráfico com 4 painéis ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(train_steps, train_loss, color="tab:blue", alpha=0.6, linewidth=0.8, label="loss treino")
    ax.plot(eval_steps, eval_loss_vals, color="tab:red", linewidth=2, marker="o", markersize=3, label="loss avaliação")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Loss de treino x avaliação")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(eval_steps, eval_chrf_vals, color="tab:green", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("step"); ax.set_ylabel("chrF++"); ax.set_title("chrF++ na avaliação")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(train_steps, train_gradnorm, color="tab:purple", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("grad_norm"); ax.set_title("Norma do gradiente")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(train_steps, train_lr, color="tab:orange", linewidth=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("taxa de aprendizado"); ax.set_title("Agendamento da taxa de aprendizado")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Histórico do treino: {ckpt_dir}", fontsize=13)
    fig.tight_layout()
    png_path = out_dir / "training_history.png"
    fig.savefig(png_path, dpi=150)
    print(f"Gráfico salvo em {png_path}")

    print(f"\n{len(train_steps)} pontos de treino, {len(eval_steps)} pontos de avaliação.")
    print(f"Final: step={train_steps[-1] if train_steps else '?'}  "
          f"train_loss={train_loss[-1] if train_loss else '?'}  "
          f"eval_loss={eval_loss_vals[-1] if eval_loss_vals else '?'}  "
          f"chrF++={eval_chrf_vals[-1] if eval_chrf_vals else '?'}")


if __name__ == "__main__":
    main()
