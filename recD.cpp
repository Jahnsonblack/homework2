#include<experimental/filesystem>
#include<fstream>
#include<iostream>

#include<string>

namespace fs=std::experimental::filesystem;
int main()
{
	using namespace std;
	string str;

	string na;

	cin >> str;
	for(auto &p: fs::recursive_directory_iterator(str))
	{
		if(!fs::is_directory(p.path()))
		{
			na = p.path();
			cout << na << endl;
		}
	}
	return 0;
}
