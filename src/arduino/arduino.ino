int laser = 1;
int motor = 2;

void setup() {
    Serial.begin(9600);
    pinMode(laser, OUTPUT);
    pinMode(motor, OUTPUT);
}

void laser_on() {
    Serial.print("Laser on!\n");
}

void laser_off() {
    Serial.print("Laser off!\n");
}

void motor_turn(int degree) {
    Serial.print("Motor turning");
}

void loop() {
    if (Serial.available()) {
        //int command = readint();
        int command = Serial.parseInt();
        if (command == -2) {
            laser_off();
        } else if (command == -1) {
            laser_on();
        } else {
            motor_turn(command);
        }
    }
}
