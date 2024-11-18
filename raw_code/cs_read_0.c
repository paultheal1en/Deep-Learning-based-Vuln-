static uint64_t cs_read (void *opaque, target_phys_addr_t addr, unsigned size)

{

    CSState *s = opaque;

    uint32_t saddr, iaddr, ret;



    saddr = addr;

    iaddr = ~0U;



    switch (saddr) {

    case Index_Address:

        ret = s->regs[saddr] & ~0x80;

        break;



    case Index_Data:

        if (!(s->dregs[MODE_And_ID] & MODE2))

            iaddr = s->regs[Index_Address] & 0x0f;

        else

            iaddr = s->regs[Index_Address] & 0x1f;



        ret = s->dregs[iaddr];

        if (iaddr == Error_Status_And_Initialization) {

            /* keep SEAL happy */

            if (s->aci_counter) {

                ret |= 1 << 5;

                s->aci_counter -= 1;

            }

        }

        break;



    default:

        ret = s->regs[saddr];

        break;

    }

    dolog ("read %d:%d -> %d\n", saddr, iaddr, ret);

    return ret;

}
