static void address_space_update_ioeventfds(AddressSpace *as)

{

    FlatRange *fr;

    unsigned ioeventfd_nb = 0;

    MemoryRegionIoeventfd *ioeventfds = NULL;

    AddrRange tmp;

    unsigned i;



    FOR_EACH_FLAT_RANGE(fr, &as->current_map) {

        for (i = 0; i < fr->mr->ioeventfd_nb; ++i) {

            tmp = addrrange_shift(fr->mr->ioeventfds[i].addr,

                                  int128_sub(fr->addr.start,

                                             int128_make64(fr->offset_in_region)));

            if (addrrange_intersects(fr->addr, tmp)) {

                ++ioeventfd_nb;

                ioeventfds = g_realloc(ioeventfds,

                                          ioeventfd_nb * sizeof(*ioeventfds));

                ioeventfds[ioeventfd_nb-1] = fr->mr->ioeventfds[i];

                ioeventfds[ioeventfd_nb-1].addr = tmp;

            }

        }

    }



    address_space_add_del_ioeventfds(as, ioeventfds, ioeventfd_nb,

                                     as->ioeventfds, as->ioeventfd_nb);



    g_free(as->ioeventfds);

    as->ioeventfds = ioeventfds;

    as->ioeventfd_nb = ioeventfd_nb;

}
