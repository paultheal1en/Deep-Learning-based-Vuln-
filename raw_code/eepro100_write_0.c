static void eepro100_write(void *opaque, target_phys_addr_t addr,

                           uint64_t data, unsigned size)

{

    EEPRO100State *s = opaque;



    switch (size) {

    case 1:

        eepro100_write1(s, addr, data);

        break;

    case 2:

        eepro100_write2(s, addr, data);

        break;

    case 4:

        eepro100_write4(s, addr, data);

        break;

    default:

        abort();

    }

}
