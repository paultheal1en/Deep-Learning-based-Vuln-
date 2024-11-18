static void end_list(Visitor *v)

{

    StringInputVisitor *siv = to_siv(v);

    siv->head = true;

}
