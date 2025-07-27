import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def compute_temporary_price_impact(prices, sizes, mid_price, side='buy', step=1):
    """
    Computes the temporary price impact g_t(x) for a given LOB snapshot.
    """
    prices = np.asarray(prices, dtype=float)
    sizes  = np.asarray(sizes, dtype=int)
    max_size = int(np.sum(sizes))
    x_vals = np.arange(0, max_size + step, step, dtype=int)
    slippages = np.zeros_like(x_vals, dtype=float)

    plateau = abs(prices[0] - mid_price)

    for i, x in enumerate(x_vals):
        if x == 0:
            slippages[i] = 0.0
        elif x <= sizes[0]:
            slippages[i] = plateau
        else:
            remaining = x
            cost = 0.0
            for price, depth in zip(prices, sizes):
                take = min(depth, remaining)
                cost += take * price
                remaining -= take
                if remaining <= 0:
                    break
            avg_price = cost / x
            slippages[i] = (avg_price - mid_price) if side=='buy' else (mid_price - avg_price)

    return x_vals, slippages

def main():
    parser = argparse.ArgumentParser(description="Compute & save temporary price impact for SOUN data")
    default_csv = os.path.join(os.path.expanduser("~"), "Downloads", "SOUN_2025-04-03 00_00_00+00_00.csv")
    parser.add_argument('--file', type=str, default=default_csv, help="Path to your SOUN CSV file")
    parser.add_argument('--step', type=int, default=1, help="Order size increment")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.file)
    df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
    snapshot = df.iloc[0]

    # Extract top 10 levels
    ask_prices = snapshot[[f'ask_px_{i:02d}' for i in range(10)]].values
    ask_sizes  = snapshot[[f'ask_sz_{i:02d}' for i in range(10)]].values
    bid_prices = snapshot[[f'bid_px_{i:02d}' for i in range(10)]].values
    bid_sizes  = snapshot[[f'bid_sz_{i:02d}' for i in range(10)]].values

    mid_price = (ask_prices[0] + bid_prices[0]) / 2.0

    # Compute impact
    x_buy, buy_imp   = compute_temporary_price_impact(ask_prices, ask_sizes, mid_price, side='buy',  step=args.step)
    x_sell, sell_imp = compute_temporary_price_impact(bid_prices, bid_sizes, mid_price, side='sell', step=args.step)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(x_buy, buy_imp,   label='Buy-Side Impact')
    plt.plot(x_sell, sell_imp, label='Sell-Side Impact')
    plt.axvline(ask_sizes[0],  color='gray', linestyle='--', label='Ask Depth Level 1')
    plt.axvline(bid_sizes[0],  color='black', linestyle='--', label='Bid Depth Level 1')
    plt.title(f"SOUN Impact vs Order Size ({snapshot['timestamp']:%Y-%m-%d %H:%M:%S})")
    plt.xlabel("Order Size (Shares)")
    plt.ylabel("Slippage ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to Downloads
    out_dir = os.path.dirname(args.file)
    out_png = os.path.join(out_dir, "soun_price_impact.png")
    plt.savefig(out_png)
    print(f"Saved graph to: {out_png}")
    plt.show()

if __name__=='__main__':
    main()
