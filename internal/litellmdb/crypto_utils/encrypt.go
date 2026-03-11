package cryptoutils

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"io"

	"golang.org/x/crypto/nacl/secretbox"
)

// EncryptValue выполняет симметричное шифрование аналогично PyNaCl SecretBox
func EncryptValue(value string, signingKey string) ([]byte, error) {
	// 1. Получаем 32-байтный ключ через SHA-256
	hash := sha256.Sum256([]byte(signingKey))
	var secretKey [32]byte
	copy(secretKey[:], hash[:])

	// 2. Генерируем случайный 24-байтный nonce
	var nonce [24]byte
	if _, err := io.ReadFull(rand.Reader, nonce[:]); err != nil {
		return nil, err
	}

	// 3. Шифруем данные.
	// secretbox.Seal добавляет зашифрованные данные К ПЕРЕДАННОМУ ПРЕФИКСУ.
	// Поэтому мы передаем nonce[:], чтобы он оказался в самом начале (как в Python).
	encrypted := secretbox.Seal(nonce[:], []byte(value), &nonce, &secretKey)

	return encrypted, nil
}

// EncryptValueHelper оборачивает результат в URL-safe Base64
func EncryptValueHelper(value string, signingKey string) (string, error) {
	// В Python: if isinstance(value, str)
	// В Go мы ожидаем string на вход, так как это типизированный язык.

	encryptedBytes, err := EncryptValue(value, signingKey)
	if err != nil {
		return "", fmt.Errorf("encryption error: %v", err)
	}

	return base64.URLEncoding.EncodeToString(encryptedBytes), nil
}
