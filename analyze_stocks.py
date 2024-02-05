import yaml
import pandas as pd
import matplotlib.pyplot as plt

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

# Show diversity in stocks
training_data_dir = params.get("training_data_directory")
company_data_file = f'{training_data_dir}/company_data.csv'
company_data = pd.read_csv(company_data_file)

industry_grouping = company_data.groupby('Industry').count()['Symbol']
sector_grouping = company_data.groupby('Sector').count()['Symbol']
state_grouping = company_data.groupby('State').count()['Symbol']
state_grouping = state_grouping.drop(['X0','X1','E9'])
country_grouping = company_data.groupby('Country').count()['Symbol']

bins = [10, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2500000]
labels = ['10-1000', '1000-5000', '5000-10000', '10000-50000', '50000-100000', '100000-500000', '500000-1000000', '1000000-2500000']
company_data['bin'] = pd.cut(company_data['Employees'], bins=bins, labels=labels)
size_grouping = company_data['bin'].value_counts().sort_index()
positions = range(len(size_grouping))

# Industry
plt.bar(industry_grouping.index, industry_grouping)
plt.xticks(industry_grouping.index, rotation=45, fontsize=2.2, ha='right')
plt.title('Stocks by Industry')
plt.xlabel('Industry Name')
plt.ylabel('Number of stocks per industry')
plt.savefig(f'{training_data_dir}/stocks_by_industry.svg', bbox_inches='tight')
plt.close()

# Sector
plt.bar(sector_grouping.index, sector_grouping)
plt.xticks(sector_grouping.index, rotation=45, fontsize=10, ha='right')
plt.title('Stocks by Sector')
plt.xlabel('Sector Name')
plt.ylabel('Number of stocks per sector')
plt.savefig(f'{training_data_dir}/stocks_by_sector.svg', bbox_inches='tight')
plt.close()

# Number of Employees
plt.bar(positions, size_grouping)
plt.xticks(positions, labels=labels, rotation=45, fontsize=5, ha='right')
plt.title('Stocks by Size')
plt.xlabel('Number of Employees')
plt.ylabel('Number of stocks per size group')
plt.savefig(f'{training_data_dir}/stocks_by_size.svg', bbox_inches='tight')
plt.close()

# Location
plt.bar(state_grouping.index, state_grouping)
plt.xticks(state_grouping.index, rotation=45, fontsize=5, ha='right')
plt.title('Stocks by State')
plt.xlabel('State Name')
plt.ylabel('Number of stocks per state')
plt.savefig(f'{training_data_dir}/stocks_by_state.svg', bbox_inches='tight')
plt.close()

plt.bar(country_grouping.index, country_grouping)
plt.xticks(country_grouping.index, rotation=45, fontsize=5, ha='right')
plt.title('Stocks by Country')
plt.xlabel('Country Name')
plt.ylabel('Number of stocks per country')
plt.savefig(f'{training_data_dir}/stocks_by_country.svg', bbox_inches='tight')
plt.close()
