#pragma once

#include <cstdlib>
#include <string>
#include <thread>
#include <chrono>

namespace cbn {

enum Pin {
    GPO_2 = 19,
    GPO_3 = 32
};

enum State {
    LOW = 0,
    HIGH = 1
};

inline void write_gpo(Pin pin, State estado) {
    std::string command = "gpioset gpiochip0 " + std::to_string(pin) + "=" + std::to_string(estado) + " &";
    std::system(command.c_str());
}

inline void fail_safe_reset() {
    std::system("gpioset gpiochip0 19=0 &");
    std::system("gpioset gpiochip0 32=0 &");
}

inline void pulse_gpo(Pin pin, int duration_ms) {
    std::thread([pin, duration_ms]() {
        write_gpo(pin, HIGH);
        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
        write_gpo(pin, LOW);
    }).detach();
}

} // namespace cbn
