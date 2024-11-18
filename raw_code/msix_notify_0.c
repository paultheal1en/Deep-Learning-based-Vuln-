void msix_notify(PCIDevice *dev, unsigned vector)

{

    MSIMessage msg;



    if (vector >= dev->msix_entries_nr || !dev->msix_entry_used[vector])

        return;

    if (msix_is_masked(dev, vector)) {

        msix_set_pending(dev, vector);

        return;

    }



    msg = msix_get_message(dev, vector);



    stl_le_phys(&address_space_memory, msg.address, msg.data);

}
