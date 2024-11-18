static void phys_page_compact_all(AddressSpaceDispatch *d, int nodes_nb)

{

    DECLARE_BITMAP(compacted, nodes_nb);



    if (d->phys_map.skip) {

        phys_page_compact(&d->phys_map, d->nodes, compacted);

    }

}
