static inline void softusb_read_pmem(MilkymistSoftUsbState *s,

        uint32_t offset, uint8_t *buf, uint32_t len)

{

    if (offset + len >= s->pmem_size) {

        error_report("milkymist_softusb: read pmem out of bounds "

                "at offset 0x%x, len %d", offset, len);


        return;

    }



    memcpy(buf, s->pmem_ptr + offset, len);

}