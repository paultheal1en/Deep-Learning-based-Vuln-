static uint8_t eepro100_read1(EEPRO100State * s, uint32_t addr)

{

    uint8_t val;

    if (addr <= sizeof(s->mem) - sizeof(val)) {

        memcpy(&val, &s->mem[addr], sizeof(val));

    }



    switch (addr) {

    case SCBStatus:

    case SCBAck:

        TRACE(OTHER, logout("addr=%s val=0x%02x\n", regname(addr), val));

        break;

    case SCBCmd:

        TRACE(OTHER, logout("addr=%s val=0x%02x\n", regname(addr), val));

#if 0

        val = eepro100_read_command(s);

#endif

        break;

    case SCBIntmask:

        TRACE(OTHER, logout("addr=%s val=0x%02x\n", regname(addr), val));

        break;

    case SCBPort + 3:

        TRACE(OTHER, logout("addr=%s val=0x%02x\n", regname(addr), val));

        break;

    case SCBeeprom:

        val = eepro100_read_eeprom(s);

        break;

    case SCBpmdr:       /* Power Management Driver Register */

        val = 0;

        TRACE(OTHER, logout("addr=%s val=0x%02x\n", regname(addr), val));

        break;

    case SCBgstat:      /* General Status Register */

        /* 100 Mbps full duplex, valid link */

        val = 0x07;

        TRACE(OTHER, logout("addr=General Status val=%02x\n", val));

        break;

    default:

        logout("addr=%s val=0x%02x\n", regname(addr), val);

        missing("unknown byte read");

    }

    return val;

}
