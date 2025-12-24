#include <windows.h>
#include <iostream>
#include <casadi/casadi.hpp>

int main()
{
    char exePath[MAX_PATH]{0};
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    std::cout << "[0] exe = " << exePath << std::endl;

    std::cout << "[1] enter main" << std::endl;

    HMODULE h = LoadLibraryA("casadi.dll");
    if (!h)
    {
        std::cout << "[E] Failed to load casadi.dll, GetLastError=" << GetLastError() << std::endl;
        return -1;
    }

    char path[MAX_PATH] = {0};
    GetModuleFileNameA(h, path, MAX_PATH);
    std::cout << "[2] Loaded casadi.dll from:\n"
              << path << std::endl;

    try
    {
        using namespace casadi;

        std::cout << "[3] build symbolic function" << std::endl;
        SX x = SX::sym("x");
        SX f = x * x + 1;
        Function F("F", {x}, {f});

        std::cout << "[4] evaluate F(3)" << std::endl;
        auto out = F(DMVector{DM(3.0)})[0];

        std::cout << "[5] F(3) = " << out << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "[EXCEPTION] std::exception: " << e.what() << std::endl;
        return -2;
    }
    catch (...)
    {
        std::cout << "[EXCEPTION] unknown" << std::endl;
        return -3;
    }

    std::cout << "[6] done" << std::endl;

    system("pause");
    return 0;
}
