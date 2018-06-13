#pragma once

#include <string>
#include <vector>

#define GREENTEXT(output) "\x1b[32;1m" << output << "\x1b[0m"
#define REDTEXT(output) "\x1b[31;1m" << output << "\x1b[0m" 
#define BLUETEXT(output) "\x1b[34;1m" << output << "\x1b[0m" 
#define YELLOWTEXT(output) "\x1b[33;1m" << output << "\x1b[0m" 


std::vector<std::string> split(const std::string& s, char delim);


std::vector<std::string> split(const std::string& s);


std::pair<std::string, std::string> splitOffDigits(std::string s);


bool endsWith(std::string str, std::string key);


std::string intToString(int number, int minLength = 0);


std::string floatToString(float number);
 

int clamp(int val, int min_val, int max_val);


std::vector<std::string> getSubPaths(std::string basePath);


std::vector<std::string> getFiles(std::string path, std::string ext, bool silent = false);