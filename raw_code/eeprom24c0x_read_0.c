static uint8_t eeprom24c0x_read(void)

{

    logout("%u: scl = %u, sda = %u, data = 0x%02x\n",

        eeprom.tick, eeprom.scl, eeprom.sda, eeprom.data);

    return eeprom.sda;

}
