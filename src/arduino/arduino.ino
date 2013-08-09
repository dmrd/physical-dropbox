#define MOTOR_PIN 3
#define LLASER_PIN 4
#define RLASER_PIN 5

void setup() {
    Serial.begin(9600);

    // set up motor control pins
    pinMode(MOTOR_PIN, OUTPUT);
    digitalWrite(MOTOR_PIN, LOW);

    pinMode(LLASER_PIN, OUTPUT);
    digitalWrite(LLASER_PIN, LOW);
    pinMode(RLASER_PIN, OUTPUT);
    digitalWrite(RLASER_PIN, LOW);
}

void llaser_on() {
    digitalWrite(LLASER_PIN, HIGH);
    Serial.print(1);
}

void rlaser_on() {
    digitalWrite(RLASER_PIN, HIGH);
    Serial.print(1);
}

void llaser_off() {
    digitalWrite(LLASER_PIN, LOW);
    Serial.print(1);
}

void rlaser_off() {
    digitalWrite(RLASER_PIN, LOW);
    Serial.print(1);
}


// rotate step motor a specific number of steps (3200 steps is a full 360 degree rotation)
void rotate(int steps) {
  int step_delay = 1500; // delay between each microstep, value between 70 and 7000 recommended (smaller means faster rotation)
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
    if (command == -4) {
      rlaser_on();
    } else if (command == -3) {
      rlaser_off();
    } else if (command == -2) {
      llaser_on();
    } else if (command == -1) {
      llaser_off();
    } else {
      rotate(command);
    }
  }
}
