#define MOTOR_PIN 3
#define LASER_PIN 4

void setup() {
    Serial.begin(9600);

    // set up motor control pins
    pinMode(MOTOR_PIN, OUTPUT);
    digitalWrite(MOTOR_PIN, LOW);
}

void laser_on() {
    digitalWrite(LASER_PIN, HIGH);
}

void laser_off() {
    digitalWrite(LASER_PIN, LOW);
}

// rotate step motor a specific number of steps (3200 steps is a full 360 degree rotation)
void rotate(int steps) {
  int step_delay = 1000; // delay between each microstep, value between 70 and 7000 recommended (smaller means faster rotation)
  for (int i = 0; i < steps; i++) {
    digitalWrite(MOTOR_PIN, HIGH);
    delayMicroseconds(step_delay);
    digitalWrite(MOTOR_PIN, LOW);
    delayMicroseconds(step_delay);
  }
  Serial.print(1);
}

void loop() {
  if (Serial.available()) {
    int command = Serial.parseInt();
    if (command == -2) {
      laser_off();
    } else if (command == -1) {
      laser_on();
    } else {
      rotate(command);
    }
  }
}
