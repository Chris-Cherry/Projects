import datetime
import numpy as np
import csv
import matplotlib.pyplot as plt


def string_to_date(string):
    try:
        split = string.decode().split('-')
    except:
        split = string.split('-')

    split = [int(i) for i in split]
    date = datetime.date(split[0], split[1], split[2])
    return date


class cramerStocks:
    """Cramer stock recommendation class to read in stock
    recommendations and market data and normalize
    performance to SNP500"""
    def __init__(self, csr_file, market_data_dir):
        self.csr_file = csr_file
        self.market_data_dir = market_data_dir
        # Using S10 to read in CSV trims the time stamp off of the
        # dates. If date format changes code must be modified to
        # process dates.
        self.csrs = np.genfromtxt(csr_file, dtype='S10', delimiter=',')

        # Translate the dates to date objects (datetime module)
        dates = [string_to_date(str) for str in self.csrs[:, 2]]
        dates = np.expand_dims(np.array(dates), axis=1)
        self.csrs = np.append(self.csrs[:, 0:2], dates, axis=1)
        self.csr_performance = []

    def process_csr(self, max_day_outlook):
        csrp_csv = open('csr_performance.csv', 'w')
        csrp_writer = csv.writer(csrp_csv)
        # Get SNP500 index data. [:, 0]=date, [:, 1]=open, [:, 2]=high
        snp_dat = np.genfromtxt(self.market_data_dir + '^GSPC.csv',
                                dtype='S10', delimiter=',')
        snp_dat = np.delete(snp_dat, (0), axis=0)
        dates = [string_to_date(str) for str in snp_dat[:, 0]]
        dates = np.expand_dims(np.array(dates), axis=1)
        snp_dat = np.append(dates, snp_dat[:, 1:], axis=1)
        # Build performance data for each csr
        for s_i, stock in enumerate(self.csrs[:, 0]):
            cl, date = self.csrs[s_i, 1:3]
            cl = cl.decode()
            stock = stock.decode()
            print(stock)
            # Build file name for market data
            market_file = self.market_data_dir + stock.lower() + '.us.txt'
            try:
                # Read stock data. [:, 0]=date, [:, 1]=open, [:, 2]=high
                stock_dat = np.genfromtxt(market_file, dtype='S10',
                                          delimiter=',')
                stock_dat = np.delete(stock_dat, (0), axis=0)
                dates = [string_to_date(str) for str in stock_dat[:, 0]]
                dates = np.expand_dims(np.array(dates), axis=1)
                stock_dat = np.append(dates, stock_dat[:, 1:], axis=1)

                start_date = date

                stock_idx = np.where(stock_dat[:, 0] == start_date)
                stock_idx = range(stock_idx[0][0], stock_idx[0][0] + max_day_outlook)

                snp_idx = np.where(snp_dat[:, 0] == start_date)
                snp_idx = range(snp_idx[0][0], snp_idx[0][0] + max_day_outlook)

                stock_subset = stock_dat[stock_idx, :]
                snp_subset = snp_dat[snp_idx, :]

                stock_subset[:, 1] = stock_subset[:, 1].astype(float)
                snp_subset[:, 1] = snp_subset[:, 1].astype(float)
                # Just take the opening price to compare stock and snp
                snp_start = snp_subset[0, 1]
                stock_start = stock_subset[0, 1]

                fc = ((stock_subset[1:, 1] - stock_start) / stock_start) / \
                     ((snp_subset[1:, 1] - snp_start) / snp_start)
                fc = list(fc)
                fc.insert(0, date)
                fc.insert(0, stock)

                if 'csr_performance' not in vars():
                    csr_performance = [fc]
                else:
                    csr_performance.append(fc)

                csrp_writer.writerow(fc)
            except:
                continue

        # We have csr_performance data as a list of lists
        csr_performance = np.array(csr_performance)
        self.csr_performance = csr_performance
        return

    def load_processed_csr(self, file):
        csr_performance = []
        csr_csv = open(file, 'r')
        csr_reader = csv.reader(csr_csv)
        for lin in csr_reader:
            csr_performance.append(lin)
        csr_performance = np.array(csr_performance)
        self.csr_performance = csr_performance
        return

    def make_graphs(self, best_day):
        nums = self.csr_performance[:, 2:].astype(float)
        sd = np.std(nums, axis=0)
        dates = [string_to_date(str) for str in self.csr_performance[:, 1]]
        means = np.mean(nums, axis=0)
        days = range(1, nums.shape[1] + 1)
        by_day = plt.figure()
        by_day_ax = by_day.add_subplot(111)
        by_day_ax.plot(days, means, '-o')
        by_day_ax.grid(color='k')
        by_day_ax.set_xlabel('Days after recommendation')
        by_day_ax.set_ylabel('Fold change to SNP500')
        by_day_ax.set_title("Cramer's Stock Recommendation Performance Over Time")
        by_day.savefig('csr_performance_by_day.pdf')

        sd_day = plt.figure()
        sd_day_ax = sd_day.add_subplot(111)
        sd_day_ax.plot(days, sd, '-o')
        sd_day_ax.grid(color='k')
        sd_day_ax.set_xlabel('Days after recommendation')
        sd_day_ax.set_ylabel('SD of Fold change to SNP500')
        sd_day_ax.set_title("SD of Cramer's Stock Recommendation Performance Over Time")
        sd_day.savefig('csr_performance_by_day_sd.pdf')

        # Plot the stock recommendations by date
        print('hi')

def main():
    cramer_stocks = cramerStocks('CSR.csv', 'data/Stocks/')
    #cramer_stocks.process_csr(max_day_outlook=60)
    cramer_stocks.load_processed_csr(file='csr_performance.csv')
    cramer_stocks.make_graphs(best_day=23)

if __name__ is '__main__':
    main()

