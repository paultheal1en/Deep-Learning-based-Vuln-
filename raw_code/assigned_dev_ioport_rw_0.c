static uint64_t assigned_dev_ioport_rw(AssignedDevRegion *dev_region,

                                       target_phys_addr_t addr, int size,

                                       uint64_t *data)

{

    uint64_t val = 0;

    int fd = dev_region->region->resource_fd;



    if (fd >= 0) {

        if (data) {

            DEBUG("pwrite data=%" PRIx64 ", size=%d, e_phys=" TARGET_FMT_plx

                  ", addr="TARGET_FMT_plx"\n", *data, size, addr, addr);

            if (pwrite(fd, data, size, addr) != size) {

                error_report("%s - pwrite failed %s",

                             __func__, strerror(errno));

            }

        } else {

            if (pread(fd, &val, size, addr) != size) {

                error_report("%s - pread failed %s",

                             __func__, strerror(errno));

                val = (1UL << (size * 8)) - 1;

            }

            DEBUG("pread val=%" PRIx64 ", size=%d, e_phys=" TARGET_FMT_plx

                  ", addr=" TARGET_FMT_plx "\n", val, size, addr, addr);

        }

    } else {

        uint32_t port = addr + dev_region->u.r_baseport;



        if (data) {

            DEBUG("out data=%" PRIx64 ", size=%d, e_phys=" TARGET_FMT_plx

                  ", host=%x\n", *data, size, addr, port);

            switch (size) {

            case 1:

                outb(*data, port);

                break;

            case 2:

                outw(*data, port);

                break;

            case 4:

                outl(*data, port);

                break;

            }

        } else {

            switch (size) {

            case 1:

                val = inb(port);

                break;

            case 2:

                val = inw(port);

                break;

            case 4:

                val = inl(port);

                break;

            }

            DEBUG("in data=%" PRIx64 ", size=%d, e_phys=" TARGET_FMT_plx

                  ", host=%x\n", val, size, addr, port);

        }

    }

    return val;

}
