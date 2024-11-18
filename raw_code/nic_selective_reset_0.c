static void nic_selective_reset(EEPRO100State * s)

{

    size_t i;

    uint16_t *eeprom_contents = eeprom93xx_data(s->eeprom);

    //~ eeprom93xx_reset(s->eeprom);

    memcpy(eeprom_contents, s->conf.macaddr.a, 6);

    eeprom_contents[0xa] = 0x4000;

    if (s->device == i82557B || s->device == i82557C)

        eeprom_contents[5] = 0x0100;

    uint16_t sum = 0;

    for (i = 0; i < EEPROM_SIZE - 1; i++) {

        sum += eeprom_contents[i];

    }

    eeprom_contents[EEPROM_SIZE - 1] = 0xbaba - sum;

    TRACE(EEPROM, logout("checksum=0x%04x\n", eeprom_contents[EEPROM_SIZE - 1]));



    memset(s->mem, 0, sizeof(s->mem));

    uint32_t val = BIT(21);

    memcpy(&s->mem[SCBCtrlMDI], &val, sizeof(val));



    assert(sizeof(s->mdimem) == sizeof(eepro100_mdi_default));

    memcpy(&s->mdimem[0], &eepro100_mdi_default[0], sizeof(s->mdimem));

}
