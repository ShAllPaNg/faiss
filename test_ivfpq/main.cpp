#include "Config.h"
#include "IndexTest.h"
#include <iostream>
#include <string>

static void PrintUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -c, --config <file>   Config file path (default: config.ini)\n";
    std::cout << "  -g, --generate        Generate default config file and exit\n";
    std::cout << "  -h, --help            Show this help message\n";
}

int main(int argc, char* argv[])
{
    std::string configPath = "config.ini";
    bool generateConfig = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "-g" || arg == "--generate") {
            generateConfig = true;
        } else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                configPath = argv[++i];
            } else {
                std::cerr << "Error: --config requires a file path\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }
    
    if (generateConfig) {
        Config defaultConfig;
        defaultConfig.SaveToFile(configPath);
        std::cout << "Default config saved to: " << configPath << std::endl;
        return 0;
    }
    
    Config config = Config::LoadFromFile(configPath);
    config.configFilePath_ = configPath;
    
    IndexTest test(config);
    test.Run();
    
    return 0;
}
