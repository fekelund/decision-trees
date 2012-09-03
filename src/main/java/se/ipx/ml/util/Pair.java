package se.ipx.ml.util;

public class Pair<L, R> {

	private final L l;
	private final R r;

	Pair(L l, R r) {
		this.l = l;
		this.r = r;
	}

	public L getLeft() {
		return l;
	}

	public R getRight() {
		return r;
	}

	public static <L, R> Pair<L, R> with(L l, R r) {
		return new Pair<L, R>(l, r);
	}

}
