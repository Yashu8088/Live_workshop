## train.py
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_and_save_models():
    print("🔹 Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape:   {y.shape}")

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print("🔹 Training CART (Gini) model...")
    cart_clf = DecisionTreeClassifier(
        criterion="gini",
        random_state=42,
        max_depth=4
    )
    cart_clf.fit(X_train, y_train)

    print("🔹 Training ID3-like (Entropy) model...")
    id3_clf = DecisionTreeClassifier(
        criterion="entropy",
        random_state=42,
        max_depth=4
    )
    id3_clf.fit(X_train, y_train)

    print("🔹 Saving models to cart_model.pkl and id3_model.pkl...")
    with open("cart_model.pkl", "wb") as f:
        pickle.dump(cart_clf, f)

    with open("id3_model.pkl", "wb") as f:
        pickle.dump(id3_clf, f)

    cart_acc = cart_clf.score(X_test, y_test)
    id3_acc = id3_clf.score(X_test, y_test)
    print(f" CART test accuracy: {cart_acc:.3f}")
    print(f"ID3  test accuracy: {id3_acc:.3f}")


if __name__ == "__main__":
    print(" Starting training script train_and_save_models()")
    train_and_save_models()
    print("Training completed.")