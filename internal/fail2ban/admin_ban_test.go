package fail2ban

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAdminBan_WildcardBansEveryModel(t *testing.T) {
	f := New(3, time.Minute, []int{500})

	ban := f.AdminBan("cred", "", time.Hour, "drain")

	assert.Equal(t, WildcardModel, ban.Model)
	assert.Equal(t, OriginAdmin, ban.Origin)
	assert.Equal(t, "admin: drain", ban.Reason)
	assert.Equal(t, 0, ban.ErrorCode)
	assert.False(t, ban.BanUntil.IsZero())

	assert.True(t, f.IsBanned("cred", "gpt-4"))
	assert.True(t, f.IsBanned("cred", "any-model-learned-later"))
	assert.True(t, f.IsBanned("cred", WildcardModel))
	assert.True(t, f.HasAnyBan("cred"))
	assert.False(t, f.IsBanned("other", "gpt-4"), "other credentials are not affected")
}

func TestAdminBan_PerModelDoesNotBanOtherModels(t *testing.T) {
	f := New(3, time.Minute, []int{500})

	f.AdminBan("cred", "gpt-4", time.Hour, "")

	assert.True(t, f.IsBanned("cred", "gpt-4"))
	assert.False(t, f.IsBanned("cred", "gpt-3"))
}

func TestAdminBan_ExplicitWildcardEqualsEmptyModel(t *testing.T) {
	f := New(3, time.Minute, []int{500})

	ban := f.AdminBan("cred", "*", time.Hour, "x")

	assert.Equal(t, WildcardModel, ban.Model)
	assert.True(t, f.IsBanned("cred", "m"))
}

func TestAdminBan_ZeroTTLIsPermanent(t *testing.T) {
	f := New(3, time.Minute, []int{500})

	ban := f.AdminBan("cred", "", 0, "until I say so")

	assert.True(t, ban.BanUntil.IsZero())
	assert.True(t, f.IsBanned("cred", "m"))
	_, finite := f.RemainingBan("cred", "m")
	assert.False(t, finite, "a permanent ban has no finite ETA")
	require.Len(t, f.GetActiveBans(), 1)
}

func TestAdminBan_ReasonPrefix(t *testing.T) {
	f := New(3, time.Minute, []int{500})

	assert.Equal(t, "admin", f.AdminBan("c", "a", time.Hour, "").Reason)
	assert.Equal(t, "admin: why", f.AdminBan("c", "b", time.Hour, "  why ").Reason)
	assert.Equal(t, "admin: kept", f.AdminBan("c", "d", time.Hour, "admin: kept").Reason)
}

func TestAdminBan_OverridesLongerAutomaticBan(t *testing.T) {
	f := New(1, 24*time.Hour, []int{500})
	f.RecordResponse("cred", "m", 500)
	require.True(t, f.IsBanned("cred", "m"))

	// BanUntil never shortens an existing ban, AdminBan replaces it.
	f.BanUntil("cred", "m", 429, time.Now().Add(time.Minute), "quota")
	ban := f.AdminBan("cred", "m", time.Minute, "override")

	assert.Equal(t, OriginAdmin, ban.Origin)
	remaining, ok := f.RemainingBan("cred", "m")
	require.True(t, ok)
	assert.LessOrEqual(t, remaining, time.Minute)
}

func TestAdminBan_ExpiresOnItsOwn(t *testing.T) {
	f := New(3, time.Minute, []int{500})

	f.AdminBan("cred", "", 30*time.Millisecond, "short")
	require.True(t, f.IsBanned("cred", "m"))

	time.Sleep(60 * time.Millisecond)
	assert.False(t, f.IsBanned("cred", "m"))
	assert.Empty(t, f.GetActiveBans())
}

func TestAdminBan_OriginOfAutomaticBans(t *testing.T) {
	f := New(1, time.Hour, []int{500})
	f.RecordResponse("a", "m", 500)
	f.BanUntil("b", "m", 429, time.Now().Add(time.Hour), "quota")

	bans := f.GetActiveBans()
	require.Len(t, bans, 2)
	assert.Equal(t, OriginFail2Ban, bans[0].Origin)
	assert.Equal(t, OriginFail2Ban, bans[1].Origin)
}

func TestRemainingBan_WildcardAndExactTakeLongest(t *testing.T) {
	f := New(3, time.Minute, []int{500})
	f.AdminBan("cred", "", time.Hour, "wide")
	f.AdminBan("cred", "m", time.Minute, "narrow")

	remaining, ok := f.RemainingBan("cred", "m")
	require.True(t, ok)
	assert.Greater(t, remaining, 30*time.Minute, "credential is usable again only after the wildcard ban lifts")

	remaining, ok = f.RemainingBan("cred", "other")
	require.True(t, ok)
	assert.Greater(t, remaining, 30*time.Minute)
}

func TestAdminUnban_EmptyModelRemovesEverything(t *testing.T) {
	f := New(3, time.Minute, []int{500})
	f.AdminBan("cred", "", time.Hour, "wide")
	f.AdminBan("cred", "m", time.Hour, "narrow")
	f.AdminBan("other", "", time.Hour, "keep")

	assert.Equal(t, 2, f.AdminUnban("cred", ""))

	assert.False(t, f.IsBanned("cred", "m"))
	assert.True(t, f.IsBanned("other", "m"), "other credentials keep their bans")
}

func TestAdminUnban_ExactModelKeepsWildcard(t *testing.T) {
	f := New(3, time.Minute, []int{500})
	f.AdminBan("cred", "", time.Hour, "wide")
	f.AdminBan("cred", "m", time.Hour, "narrow")

	assert.Equal(t, 1, f.AdminUnban("cred", "m"))
	assert.True(t, f.IsBanned("cred", "m"), "wildcard ban still applies")

	assert.Equal(t, 1, f.AdminUnban("cred", WildcardModel))
	assert.False(t, f.IsBanned("cred", "m"))
}

func TestAdminUnban_IsIdempotent(t *testing.T) {
	f := New(3, time.Minute, []int{500})

	assert.Equal(t, 0, f.AdminUnban("cred", ""))
	assert.Equal(t, 0, f.AdminUnban("cred", "m"))

	f.AdminBan("cred", "", time.Hour, "x")
	assert.Equal(t, 1, f.AdminUnban("cred", ""))
	assert.Equal(t, 0, f.AdminUnban("cred", ""))
}

func TestAdminUnban_LiftsAutomaticBansToo(t *testing.T) {
	f := New(1, time.Hour, []int{500})
	f.RecordResponse("cred", "m", 500)
	require.True(t, f.IsBanned("cred", "m"))

	assert.Equal(t, 1, f.AdminUnban("cred", "m"))
	assert.False(t, f.IsBanned("cred", "m"))
}

func TestGetActiveBans_SortedAndSkipsExpired(t *testing.T) {
	f := New(3, time.Minute, []int{500})
	f.AdminBan("b", "z", time.Hour, "")
	f.AdminBan("a", "y", time.Hour, "")
	f.AdminBan("a", "x", time.Hour, "")
	f.AdminBan("gone", "m", time.Millisecond, "")
	time.Sleep(20 * time.Millisecond)

	bans := f.GetActiveBans()
	require.Len(t, bans, 3)
	assert.Equal(t, []string{"a|x", "a|y", "b|z"}, []string{
		bans[0].Credential + "|" + bans[0].Model,
		bans[1].Credential + "|" + bans[1].Model,
		bans[2].Credential + "|" + bans[2].Model,
	})
}
