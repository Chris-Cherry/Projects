import datetime
import numpy as np


def string_to_date(string):
    split = string.decode().split('-')
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

    def process_csr(self, max_day_outlook):
        csr_performance = np.zeros(shape=(self.csrs.shape[0], max_day_outlook + 2))
        csr_performance[:] = np.nan
        # Get SNP500 index data. [:, 1]
        snp_dat = np.genfromtxt(self.market_data_dir + '^GSPC.csv', delimiter=',')
        snp_dat = np.delete(np.delete(snp_dat, 0), 1)
        for s_i, stock in enumerate(self.csrs[:, 0]):
            cl, date = self.csrs[s_i, 1:3]
            cl = cl.decode()
            stock = stock.decode()
            # Build file name for market data
            market_file = self.market_data_dir + stock + '.us.txt'



def main():
    cramer_stocks = cramerStocks('CSR.csv', 'data/Stocks/')
    cramer_stocks.process_csr(max_day_outlook=60)


if __name__ is '__main__':
    main()

