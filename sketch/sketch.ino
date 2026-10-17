// SPDX-License-Identifier: MPL-2.0

#include <Arduino_RouterBridge.h>
#include <Wire.h>

// PCA9685 / HW-170 settings.
constexpr uint8_t PCA9685_ADDRESS = 0x40;
constexpr uint8_t PCA9685_MODE1 = 0x00;
constexpr uint8_t PCA9685_PRESCALE = 0xFE;
constexpr uint8_t LED0_ON_L = 0x06;

// These are the calibrated limits from the original servo test.
// Motor 0: base / horizontal pan.
#define SERVO0_MIN 110
#define SERVO0_MAX 500
// Motor 1: upper bracket / vertical camera tilt.
#define SERVO1_MIN 110
#define SERVO1_MAX 200

constexpr uint8_t PAN_CHANNEL = 0;
constexpr uint8_t TILT_CHANNEL = 1;
constexpr uint16_t PAN_START = (SERVO0_MIN + SERVO0_MAX) / 2;
constexpr uint16_t TILT_START = (SERVO1_MIN + SERVO1_MAX) / 2;

// Match the fast but smooth reaction in the reference tracker.  A new pulse is
// written every 20 ms (one 50 Hz servo frame), with responsive independent
// slew rates for the wider pan range and the narrower tilt range.
constexpr uint16_t CONTROL_INTERVAL_MS = 20;
constexpr uint16_t PAN_STEP_PER_FRAME = 22;
constexpr uint16_t TILT_STEP_PER_FRAME = 10;

volatile uint16_t panTarget = PAN_START;
volatile uint16_t tiltTarget = TILT_START;
uint16_t panPosition = PAN_START;
uint16_t tiltPosition = TILT_START;
uint32_t lastControlAt = 0;

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(PCA9685_ADDRESS);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(PCA9685_ADDRESS);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(PCA9685_ADDRESS, static_cast<uint8_t>(1));
  return Wire.available() ? Wire.read() : 0;
}

void writeServoPulse(uint8_t channel, uint16_t pulse) {
  const uint8_t reg = LED0_ON_L + 4 * channel;
  Wire.beginTransmission(PCA9685_ADDRESS);
  Wire.write(reg);
  Wire.write(0);                 // ON low byte
  Wire.write(0);                 // ON high byte
  Wire.write(pulse & 0xFF);      // OFF low byte
  Wire.write((pulse >> 8) & 0x0F); // OFF high byte
  Wire.endTransmission();
}

void initializePca9685() {
  writeRegister(PCA9685_MODE1, 0x00);
  delay(10);
  const uint8_t oldMode = readRegister(PCA9685_MODE1);
  writeRegister(PCA9685_MODE1, (oldMode & 0x7F) | 0x10); // sleep
  writeRegister(PCA9685_PRESCALE, 121);                 // approximately 50 Hz
  writeRegister(PCA9685_MODE1, oldMode);
  delay(5);
  writeRegister(PCA9685_MODE1, oldMode | 0xA1); // restart, auto-increment, all-call
}

uint16_t moveToward(uint16_t current, uint16_t target, uint16_t maximumStep) {
  if (current < target) {
    return static_cast<uint16_t>(min(static_cast<uint32_t>(target),
                                     static_cast<uint32_t>(current) + maximumStep));
  }
  if (current > target) {
    const uint16_t difference = current - target;
    return difference <= maximumStep ? target : current - maximumStep;
  }
  return current;
}

// Called from Python through RouterBridge.  Keeping this callback short avoids
// blocking the bridge; the loop below performs all gradual motor movement.
void setFaceTarget(int pan, int tilt) {
  panTarget = constrain(pan, SERVO0_MIN, SERVO0_MAX);
  tiltTarget = constrain(tilt, SERVO1_MIN, SERVO1_MAX);
}

void setup() {
  Wire.begin();
  initializePca9685();
  writeServoPulse(PAN_CHANNEL, panPosition);
  writeServoPulse(TILT_CHANNEL, tiltPosition);

  Bridge.begin();
  Bridge.provide("set_face_target", setFaceTarget);
}

void loop() {
  const uint32_t now = millis();
  if (now - lastControlAt < CONTROL_INTERVAL_MS) {
    return;
  }
  lastControlAt = now;

  const uint16_t requestedPan = panTarget;
  const uint16_t requestedTilt = tiltTarget;
  const uint16_t nextPan = moveToward(panPosition, requestedPan, PAN_STEP_PER_FRAME);
  const uint16_t nextTilt = moveToward(tiltPosition, requestedTilt, TILT_STEP_PER_FRAME);

  if (nextPan != panPosition) {
    panPosition = nextPan;
    writeServoPulse(PAN_CHANNEL, panPosition);
  }
  if (nextTilt != tiltPosition) {
    tiltPosition = nextTilt;
    writeServoPulse(TILT_CHANNEL, tiltPosition);
  }
}
