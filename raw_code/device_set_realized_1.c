static void device_set_realized(Object *obj, bool value, Error **errp)
{
    DeviceState *dev = DEVICE(obj);
    DeviceClass *dc = DEVICE_GET_CLASS(dev);
    HotplugHandler *hotplug_ctrl;
    BusState *bus;
    Error *local_err = NULL;
    bool unattached_parent = false;
    static int unattached_count;
    int ret;
    if (dev->hotplugged && !dc->hotpluggable) {
        error_setg(errp, QERR_DEVICE_NO_HOTPLUG, object_get_typename(obj));
        return;
    if (value && !dev->realized) {
        if (!obj->parent) {
            gchar *name = g_strdup_printf("device[%d]", unattached_count++);
            object_property_add_child(container_get(qdev_get_machine(),
                                                    "/unattached"),
                                      name, obj, &error_abort);
            unattached_parent = true;
            g_free(name);
        hotplug_ctrl = qdev_get_hotplug_handler(dev);
        if (hotplug_ctrl) {
            hotplug_handler_pre_plug(hotplug_ctrl, dev, &local_err);
            if (local_err != NULL) {
        if (dc->realize) {
            dc->realize(dev, &local_err);
        if (local_err != NULL) {
        DEVICE_LISTENER_CALL(realize, Forward, dev);
        if (hotplug_ctrl) {
            hotplug_handler_plug(hotplug_ctrl, dev, &local_err);
        if (local_err != NULL) {
            goto post_realize_fail;
        if (qdev_get_vmsd(dev)) {
            if (vmstate_register_with_alias_id(dev, -1, qdev_get_vmsd(dev), dev,
                                               dev->instance_id_alias,
                                               dev->alias_required_for_version,
                                               &local_err) < 0) {
                goto post_realize_fail;
        QLIST_FOREACH(bus, &dev->child_bus, sibling) {
            object_property_set_bool(OBJECT(bus), true, "realized",
                                         &local_err);
            if (local_err != NULL) {
                goto child_realize_fail;
        if (dev->hotplugged) {
            device_reset(dev);
        dev->pending_deleted_event = false;
    } else if (!value && dev->realized) {
        Error **local_errp = NULL;
        QLIST_FOREACH(bus, &dev->child_bus, sibling) {
            local_errp = local_err ? NULL : &local_err;
            object_property_set_bool(OBJECT(bus), false, "realized",
                                     local_errp);
        if (qdev_get_vmsd(dev)) {
            vmstate_unregister(dev, qdev_get_vmsd(dev), dev);
        if (dc->unrealize) {
            local_errp = local_err ? NULL : &local_err;
            dc->unrealize(dev, local_errp);
        dev->pending_deleted_event = true;
        DEVICE_LISTENER_CALL(unrealize, Reverse, dev);
    if (local_err != NULL) {
    dev->realized = value;
    return;
child_realize_fail:
    QLIST_FOREACH(bus, &dev->child_bus, sibling) {
        object_property_set_bool(OBJECT(bus), false, "realized",
                                 NULL);
    if (qdev_get_vmsd(dev)) {
        vmstate_unregister(dev, qdev_get_vmsd(dev), dev);
post_realize_fail:
    if (dc->unrealize) {
        dc->unrealize(dev, NULL);
fail:
    error_propagate(errp, local_err);
    if (unattached_parent) {
        object_unparent(OBJECT(dev));
        unattached_count--;