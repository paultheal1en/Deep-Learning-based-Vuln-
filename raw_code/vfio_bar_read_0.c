static uint64_t vfio_bar_read(void *opaque,

                              hwaddr addr, unsigned size)

{

    VFIOBAR *bar = opaque;

    union {

        uint8_t byte;

        uint16_t word;

        uint32_t dword;

        uint64_t qword;

    } buf;

    uint64_t data = 0;



    if (pread(bar->fd, &buf, size, bar->fd_offset + addr) != size) {

        error_report("%s(,0x%"HWADDR_PRIx", %d) failed: %m",

                     __func__, addr, size);

        return (uint64_t)-1;

    }



    switch (size) {

    case 1:

        data = buf.byte;

        break;

    case 2:

        data = le16_to_cpu(buf.word);

        break;

    case 4:

        data = le32_to_cpu(buf.dword);

        break;

    default:

        hw_error("vfio: unsupported read size, %d bytes\n", size);

        break;

    }



    DPRINTF("%s(BAR%d+0x%"HWADDR_PRIx", %d) = 0x%"PRIx64"\n",

            __func__, bar->nr, addr, size, data);



    /* Same as write above */

    vfio_eoi(container_of(bar, VFIODevice, bars[bar->nr]));



    return data;

}
