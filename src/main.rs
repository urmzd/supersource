struct Node<DataType> {
    data: DataType,
    next: Box<Node<DataType>>,
}

fn main() {
    println!("Hello, world!");
}
