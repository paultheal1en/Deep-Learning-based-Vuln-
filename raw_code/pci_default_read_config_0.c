uint32_t pci_default_read_config(PCIDevice *d,

                                 uint32_t address, int len)

{

    uint32_t val;



    switch(len) {

    default:

    case 4:

	if (address <= 0xfc) {

            val = pci_get_long(d->config + address);

	    break;

	}

	/* fall through */

    case 2:

        if (address <= 0xfe) {

            val = pci_get_word(d->config + address);

	    break;

	}

	/* fall through */

    case 1:

        val = pci_get_byte(d->config + address);

        break;

    }

    return val;

}
