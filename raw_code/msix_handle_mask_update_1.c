static void msix_handle_mask_update(PCIDevice *dev, int vector)

{

    if (!msix_is_masked(dev, vector) && msix_is_pending(dev, vector)) {

        msix_clr_pending(dev, vector);

        msix_notify(dev, vector);

    }

}
