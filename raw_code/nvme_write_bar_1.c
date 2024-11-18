static void nvme_write_bar(NvmeCtrl *n, hwaddr offset, uint64_t data,

    unsigned size)

{

    switch (offset) {

    case 0xc:

        n->bar.intms |= data & 0xffffffff;

        n->bar.intmc = n->bar.intms;

        break;

    case 0x10:

        n->bar.intms &= ~(data & 0xffffffff);

        n->bar.intmc = n->bar.intms;

        break;

    case 0x14:

        /* Windows first sends data, then sends enable bit */

        if (!NVME_CC_EN(data) && !NVME_CC_EN(n->bar.cc) &&

            !NVME_CC_SHN(data) && !NVME_CC_SHN(n->bar.cc))

        {

            n->bar.cc = data;

        }



        if (NVME_CC_EN(data) && !NVME_CC_EN(n->bar.cc)) {

            n->bar.cc = data;

            if (nvme_start_ctrl(n)) {

                n->bar.csts = NVME_CSTS_FAILED;

            } else {

                n->bar.csts = NVME_CSTS_READY;

            }

        } else if (!NVME_CC_EN(data) && NVME_CC_EN(n->bar.cc)) {

            nvme_clear_ctrl(n);

            n->bar.csts &= ~NVME_CSTS_READY;

        }

        if (NVME_CC_SHN(data) && !(NVME_CC_SHN(n->bar.cc))) {

                nvme_clear_ctrl(n);

                n->bar.cc = data;

                n->bar.csts |= NVME_CSTS_SHST_COMPLETE;

        } else if (!NVME_CC_SHN(data) && NVME_CC_SHN(n->bar.cc)) {

                n->bar.csts &= ~NVME_CSTS_SHST_COMPLETE;

                n->bar.cc = data;

        }

        break;

    case 0x24:

        n->bar.aqa = data & 0xffffffff;

        break;

    case 0x28:

        n->bar.asq = data;

        break;

    case 0x2c:

        n->bar.asq |= data << 32;

        break;

    case 0x30:

        n->bar.acq = data;

        break;

    case 0x34:

        n->bar.acq |= data << 32;

        break;

    default:

        break;

    }

}
