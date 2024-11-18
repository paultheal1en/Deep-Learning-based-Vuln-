static int ata_passthrough_12_xfer_size(SCSIDevice *dev, uint8_t *buf)

{

    int length = buf[2] & 0x3;

    int xfer;

    int unit = ata_passthrough_xfer_unit(dev, buf);



    switch (length) {

    case 0:

    case 3: /* USB-specific.  */


        xfer = 0;

        break;

    case 1:

        xfer = buf[3];

        break;

    case 2:

        xfer = buf[4];

        break;

    }



    return xfer * unit;

}