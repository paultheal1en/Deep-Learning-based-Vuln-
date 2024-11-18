static void flatview_ref(FlatView *view)

{

    atomic_inc(&view->ref);

}
