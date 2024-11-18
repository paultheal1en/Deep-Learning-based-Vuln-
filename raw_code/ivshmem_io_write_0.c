static void ivshmem_io_write(void *opaque, target_phys_addr_t addr,

                             uint64_t val, unsigned size)

{

    IVShmemState *s = opaque;



    uint16_t dest = val >> 16;

    uint16_t vector = val & 0xff;



    addr &= 0xfc;



    IVSHMEM_DPRINTF("writing to addr " TARGET_FMT_plx "\n", addr);

    switch (addr)

    {

        case INTRMASK:

            ivshmem_IntrMask_write(s, val);

            break;



        case INTRSTATUS:

            ivshmem_IntrStatus_write(s, val);

            break;



        case DOORBELL:

            /* check that dest VM ID is reasonable */

            if (dest > s->max_peer) {

                IVSHMEM_DPRINTF("Invalid destination VM ID (%d)\n", dest);

                break;

            }



            /* check doorbell range */

            if (vector < s->peers[dest].nb_eventfds) {

                IVSHMEM_DPRINTF("Notifying VM %d on vector %d\n", dest, vector);

                event_notifier_set(&s->peers[dest].eventfds[vector]);

            }

            break;

        default:

            IVSHMEM_DPRINTF("Invalid VM Doorbell VM %d\n", dest);

    }

}
