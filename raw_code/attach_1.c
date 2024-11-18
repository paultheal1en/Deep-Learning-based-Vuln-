static void attach(sPAPRDRConnector *drc, DeviceState *d, void *fdt,
                   int fdt_start_offset, bool coldplug, Error **errp)
{
    DPRINTFN("drc: %x, attach", get_index(drc));
    if (drc->isolation_state != SPAPR_DR_ISOLATION_STATE_ISOLATED) {
        error_setg(errp, "an attached device is still awaiting release");
        return;
    if (drc->type == SPAPR_DR_CONNECTOR_TYPE_PCI) {
        g_assert(drc->allocation_state == SPAPR_DR_ALLOCATION_STATE_USABLE);
    g_assert(fdt || coldplug);
    /* NOTE: setting initial isolation state to UNISOLATED means we can't
     * detach unless guest has a userspace/kernel that moves this state
     * back to ISOLATED in response to an unplug event, or this is done
     * manually by the admin prior. if we force things while the guest
     * may be accessing the device, we can easily crash the guest, so we
     * we defer completion of removal in such cases to the reset() hook.
     */
    if (drc->type == SPAPR_DR_CONNECTOR_TYPE_PCI) {
        drc->isolation_state = SPAPR_DR_ISOLATION_STATE_UNISOLATED;
    drc->indicator_state = SPAPR_DR_INDICATOR_STATE_ACTIVE;
    drc->dev = d;
    drc->fdt = fdt;
    drc->fdt_start_offset = fdt_start_offset;
    drc->configured = coldplug;
    /* 'logical' DR resources such as memory/cpus are in some cases treated
     * as a pool of resources from which the guest is free to choose from
     * based on only a count. for resources that can be assigned in this
     * fashion, we must assume the resource is signalled immediately
     * since a single hotplug request might make an arbitrary number of
     * such attached resources available to the guest, as opposed to
     * 'physical' DR resources such as PCI where each device/resource is
     * signalled individually.
     */
    drc->signalled = (drc->type != SPAPR_DR_CONNECTOR_TYPE_PCI)
                     ? true : coldplug;
    object_property_add_link(OBJECT(drc), "device",
                             object_get_typename(OBJECT(drc->dev)),
                             (Object **)(&drc->dev),
                             NULL, 0, NULL);