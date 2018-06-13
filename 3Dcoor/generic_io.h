#pragma once

#include <fstream>

//»ù±¾¶ÁÐ´²Ù×÷= =. 

namespace jp
{
    template<class T>
    void write(std::ofstream& file, const T& b)
    {
	file.write((char*) &b, sizeof(T));
    }
    
    template<class T>
    void read(std::ifstream& file, T& b)
    {
	file.read(reinterpret_cast<char*>(&b), sizeof(T));
    }

    template<class T>
    void write(std::ofstream& file, const std::vector<T>& v)
    {
	write<unsigned>(file, v.size());
	for(unsigned i = 0; i < v.size(); i++)
	    write(file, v[i]);
    }

    template<class T>
    void read(std::ifstream& file, std::vector<T>& v)
    {
	unsigned size;
	read<unsigned>(file, size);
	v.resize(size);
	for(unsigned i = 0; i < v.size(); i++)
	{
	    read(file, v[i]);
	}
    }

    template<class T1, class T2>
    void write(std::ofstream& file, const std::map<T1, T2>& m)
    {
	write<unsigned>(file, m.size());
	for(typename std::map<T1, T2>::const_iterator it = m.begin(); it != m.end(); it++)
	{
	    write(file, it->first);
	    write(file, it->second);
	}
    }

    template<class T1, class T2>
    void read(std::ifstream& file, std::map<T1, T2>& m)
    {
	unsigned size;
	T1 key;
	T2 value;
	read<unsigned>(file, size);
	for(unsigned i = 0; i < size; i++)
	{
	    read(file, key);
	    read(file, value);
	    m[key] = value;
	}
    }    
    
    template<class T>
    void write(std::ofstream& file, const cv::Mat_<T>& m)
    {
	write<int>(file, m.rows);
	write<int>(file, m.cols);      
	for(unsigned i = 0; i < m.rows; i++)
	for(unsigned j = 0; j < m.cols; j++)
	    write(file, m(i, j));
    }
    
    template<class T>
    void read(std::ifstream& file, cv::Mat_<T>& m)
    {
	int rows, cols;
	read<int>(file, rows);
	read<int>(file, cols);
	m = cv::Mat_<T>(rows, cols);
	for(unsigned i = 0; i < rows; i++)
	for(unsigned j = 0; j < cols; j++)
	    read(file, m(i, j));
    }
    
    template<class T, int dim>
    void write(std::ofstream& file, const cv::Vec<T, dim>& v)
    {
	for(unsigned i = 0; i < dim; i++)
	    write(file, v[i]);
    }

    template<class T, int dim>
    void read(std::ifstream& file, const cv::Vec<T, dim>& v)
    {
	for(unsigned i = 0; i < dim; i++)
	    read(file, v[i]);
    }

    template<class T>
    void write(std::string& fileName, const T& b)
    {
        std::ofstream file;
	file.open(fileName, std::ofstream::binary);  
	jp::write(file, b);
	file.close();
    }
}