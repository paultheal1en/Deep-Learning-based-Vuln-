uint16_t eeprom93xx_read(eeprom_t *eeprom)

{

    /* Return status of pin DO (0 or 1). */

    logout("CS=%u DO=%u\n", eeprom->eecs, eeprom->eedo);

    return (eeprom->eedo);

}
