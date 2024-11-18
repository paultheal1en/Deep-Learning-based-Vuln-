static QmpOutputVisitor *to_qov(Visitor *v)

{

    return container_of(v, QmpOutputVisitor, visitor);

}
