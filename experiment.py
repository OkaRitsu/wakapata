import csv

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, init_w0, init_w1):
        # 重みの初期化
        self.params = {}
        self.params['w0'] = init_w0
        self.params['w1'] = init_w1
        # 重みの変化を記録するための配列
        self.w1_list = [init_w1]
        self.w0_list = [init_w0]
        # 損失関数の変化を記録するための配列
        self.losses = []

    def predict(self, x):
        """
        識別信号g(x_i)

        引数:
            x: 学習パターンの配列
        返り値：
            y: 識別関数の出力の配列
        """
        w0, w1 = self.params['w0'], self.params['w1']
        y = w0 + w1 * x
        return y

    def loss(self, x, b):
        """
        二乗誤差J

        引数：
            x: 学習パターンの配列
            b: 教師信号の配列
        返り値：
            loss: 二乗誤差
        """
        y = self.predict(x)
        loss = 0.5 * np.sum((y - b) ** 2)
        return loss

    def numerical_gradient(self, x, b):
        """
        二乗誤差を重みベクトルで微分する（中心差分公式）

        引数：
            x: 学習パターンの配列
            b: 教師信号の配列
        返り値：
            grads:
                勾配を記録した辞書
        """
        grads = {}
        # w0とw1でそれぞれ微分する
        for key in ('w0', 'w1'):
            tmp_w = self.params[key]
            h = 1e-4

            # w+hでlossを計算
            self.params[key] = tmp_w + h
            loss1 = self.loss(x, b)
            # w-hでlossを計算
            self.params[key] = tmp_w - h
            loss2 = self.loss(x, b)
            # 勾配を計算
            grads[key] = (loss1 - loss2) / (2 * h)
            # 重みをもとに戻す
            self.params[key] = tmp_w
        return grads

    def train(self, x, b, lr, boader, batch_size):
        """
        ウィドローホフの学習規則に従って重みを修正する

        引数：
            x: 学習パターンの配列
            b: 教師信号の配列
            lr: 学習係数
            boader: 評価関数が収束したと判断するためのしきい値
            batch_size: バッチサイズ．１のときはオンライン学習，len(x)のときはバッチ学習
        """

        # 記録用のcsvファイル用意
        filename = f'./results/lr-{lr}_boader-{boader}_batchsize-{batch_size}.csv'
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'w0', 'w1', 'loss'])

        # 収束しなかった場合ここで指定したエポックで重み修正をやめる
        max_epoch = 1000

        # ロスの変化を求めるため
        tmp_loss = 9999999

        for epoch in range(max_epoch):
            for step in range(0, len(x), batch_size):
                x_batch = x[step: step+batch_size]
                b_batch = b[step: step+batch_size]
                grad = self.numerical_gradient(x_batch, b_batch)

                # パラメータ更新
                for key in ('w0', 'w1'):
                    self.params[key] -= lr * grad[key] / batch_size

                loss = self.loss(x, b)
                # print('loss: ', loss)
                self.losses.append(loss)
                self.w0_list.append(self.params['w0'])
                self.w1_list.append(self.params['w1'])

                with open(filename, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, step, self.params['w0'], self.params['w1'], loss])

            # 評価関数の変化がしきい値を下回ったらやめる
            diff_loss = abs(tmp_loss - loss)
            tmp_loss = loss
            print(f'{epoch+1}epoch', diff_loss)
            if diff_loss < boader:
                # print(self.params)
                return

            # print(f'{epoch}epoch {step}steps', self.params)


def wakapata_experiment(x, b):
    """
    わかパタp.51の実験を行う

    引数：
        x: 学習パターンの配列
        b: 教師信号の配列
    返り値：
        fig: わかパタp.52の図3.2,3.3を横に並べた図
    """
    # 初期値
    w0 = 11.0
    w1 = 5.0
    # 学習率
    lr = 0.1
    # 評価関数が収束すると判断するためのしきい値（わかパタでは0.01）
    boader = 0.01
    # bach_size=1でオンライン学習，batch_size = len(x)でバッチ学習
    batch_sizes = [len(x), 1]
    labels = ['batch', 'online']

    # 描画の準備
    fig = plt.figure(figsize=(10, 5))

    # 図3.2
    ax1 = fig.add_subplot(1, 2, 1)
    # 初期値
    ax1.scatter(w1, w0, marker='x', c='k')
    ax1.annotate(f'({w1}, {w0})', xy=(w1, w0))
    # 最適値
    ax1.scatter(0.804, 0.241, c='k')
    ax1.annotate('(0.804, 0.241)', xy=(0.804, 0.241))
    # 軸の設定（わかパタ図3.2と揃えている）
    ax1.set_xlim(-10, 10)
    ax1.set_xlabel('w1')
    ax1.set_ylim(-8, 12)
    ax1.set_ylabel('w0')

    # 図3.3
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('重み修正回数')
    ax2.set_ylabel('評価関数J(w)')

    # オンライン学習とバッチ学習を実施
    for batch_size, label in zip(batch_sizes, labels):
        perceptron = Perceptron(w0, w1)
        perceptron.train(x, b, lr, boader, batch_size)
        # 重みの変化
        ax1.plot(perceptron.w1_list, perceptron.w0_list, label=label)
        # 評価関数の変化
        ax2.plot(perceptron.losses, label=label)

    # 図3.2の等高線等高線
    x1 = np.linspace(-10, 10, 200)
    x2 = np.linspace(-8, 12, 200)
    z = np.array([[0.5 * np.sum((w0 + w1 * x - b) ** 2) for w1 in x1] for w0 in x2])
    ax1.contour(x1, x2, z, colors='black', levels=20)
    ax1.set_aspect('equal')

    return fig


def experiment(x, b, w0=11.0, w1=5.0, lr=0.1, batch_size=1, boader=0.01):
    """
    わかパタとは違う実験を行って比較する．
    引数で指定しないと，わかパタと同じ条件（オンライン学習）になる．

    引数:
        x: 学習パターンの配列
        b: 教師信号の配列
        w0: w0の初期値
        w1: w1の初期値
        lr: 学習率
        batch_size: バッチサイズ，１のときはオンライン学習，6のときはバッチ学習
        boader: lossが収束したと判断するためのしきち値
    """
    # わかパタの実験結果を取得して上に重ねる
    fig = wakapata_experiment(x, b)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    perceptron = Perceptron(w0, w1)
    perceptron.train(x, b, lr, boader, batch_size)

    # 初期値
    ax1.scatter(w1, w0, marker='x', c='k')
    # 重みの変化
    ax1.plot(perceptron.w1_list, perceptron.w0_list, label='original experience')
    # 評価関数の変化
    ax2.plot(perceptron.losses, label='original experience')

    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='upper center')
    plt.legend()
    plt.show()


def main():
    # 学習パターン
    x = np.array([1.2, 0.2, -0.2, -0.5, -1.0, -1.5])
    # 教師データ
    b = np.array([1, 1, -1, 1, -1, -1])

    experiment(x, b, boader=0.01)


if __name__ == '__main__':
    main()
